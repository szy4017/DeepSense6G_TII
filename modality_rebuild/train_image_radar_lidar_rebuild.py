import argparse
import json
import os, sys
import csv
import time

from tqdm import tqdm
import pandas as pd
from datetime import datetime
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
torch.backends.cudnn.benchmark = True
from scheduler import CyclicCosineDecayLR

from config_seq import GlobalConfig
from data2_seq import CARLA_Data

import torchvision

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0]/255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1]/255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2]/255.0 - 0.406) / 0.225
    return x

def compute_acc(y_pred, y_true, top_k=[1,2,3]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx])
            total_hits[k_idx] += 1 if hit else 0
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true)*100, 4)

def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
	"""
    The top-k MBD (Minimum Beam Distance) as the minimum distance
    of any beam in the top-k set of predicted beams to the ground truth beam.
    Then we take the average across all samples.
    Then we average that number over all the considered Ks.
    """
	n_samples = y_pred.shape[0]
	yk = np.zeros(max_k)
	for k in range(max_k):
		acc_avg_min_beam_dist = 0
		idxs_up_to_k = np.arange(k + 1)
		for i in range(n_samples):
			aux1 = np.abs(y_pred[i, idxs_up_to_k] - y_true[i]) / delta
			# Compute min between beam diff and 1
			aux2 = np.min(np.stack((aux1, np.zeros_like(aux1) + 1), axis=0), axis=0)
			acc_avg_min_beam_dist += np.min(aux2)

		yk[k] = 1 - acc_avg_min_beam_dist / n_samples

	return np.mean(yk)

class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        B, C, len = x1.size()
        x1 = torch.sum(x1, dim=-1)
        x2 = torch.sum(x2, dim=-1)
        B = int(B/5)
        x1 = x1.view(B, -1)
        x2 = x2.view(B, -1)

        z_i = F.normalize(x1, dim=1)
        z_j = F.normalize(x2, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, B)
        sim_ji = torch.diag(similarity_matrix, -B)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        negatives_mask = (~torch.eye(B * 2, B * 2, dtype=bool).to(x1.device)).float()
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * B)
        return loss

class ImageEncoder(nn.Module):
    """
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim=512, normalize=True, weight=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(weights=True)
        self.features.fc = nn.Sequential()
        if weight:
            self.load_pretrained_weight()

    def load_pretrained_weight(self):
        model_dict = torch.load('image_encoder.pth')
        self.load_state_dict(model_dict)
        print('Pretrained weights loaded from image_encoder.pth')

    def forward(self, inputs):
        # c = 0
        # for x in inputs:
        #     if self.normalize:
        #         x = normalize_imagenet(x)
        #     c += self.features(x)
        # return c
        image_feat_l1 = self.features.conv1(inputs)
        image_feat_l1 = self.features.bn1(image_feat_l1)
        image_feat_l1 = self.features.relu(image_feat_l1)
        image_feat_l1 = self.features.maxpool(image_feat_l1)  # (bz*seq_len, 64, 64, 64)
        image_feat_l1 = self.features.layer1(image_feat_l1)
        return image_feat_l1

class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=1, weight=True):
        super().__init__()
        self._model = models.resnet18(weights=True)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        if weight:
            self.load_pretrained_weight()

    def load_pretrained_weight(self):
        model_dict = torch.load('lidar_encoder.pth')
        self.load_state_dict(model_dict)
        print('Pretrained weights loaded from lidar_encoder.pth')

    def forward(self, inputs):
        # features = 0
        # for lidar_data in inputs:
        #     lidar_feature = self._model(lidar_data)
        #     features += lidar_feature
        # return features
        lidar_feat_l1 = self._model.conv1(inputs)
        lidar_feat_l1 = self._model.bn1(lidar_feat_l1)
        lidar_feat_l1 = self._model.relu(lidar_feat_l1)
        lidar_feat_l1 = self._model.maxpool(lidar_feat_l1)  # (bz*seq_len, 64, 64, 64)
        lidar_feat_l1 = self._model.layer1(lidar_feat_l1)
        return lidar_feat_l1

class RadarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2, weight=True):
        super().__init__()
        self._model = models.resnet18(weights=True)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        if weight:
            self.load_pretrained_weight()

    def load_pretrained_weight(self):
        model_dict = torch.load('radar_encoder.pth')
        self.load_state_dict(model_dict)
        print('Pretrained weights loaded from radar_encoder.pth')

    def forward(self, inputs):
        # features = 0
        # for lidar_data in inputs:
        #     lidar_feature = self._model(lidar_data)
        #     features += lidar_feature
        # return features
        radar_feat_l1 = self._model.conv1(inputs)
        radar_feat_l1 = self._model.bn1(radar_feat_l1)
        radar_feat_l1 = self._model.relu(radar_feat_l1)
        radar_feat_l1 = self._model.maxpool(radar_feat_l1)  # (bz*seq_len, 64, 64, 64)
        radar_feat_l1 = self._model.layer1(radar_feat_l1)
        return radar_feat_l1

# class FusionEncoder(nn.Module):
#     def __init__(self, input_dim=1536, out_dim=64, hidden=512):
#         super().__init__()
#         self.enc_net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(hidden, out_dim)
#         )
#
#     def forward(self, feat):
#         return self.enc_net(feat)

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        )

    def forward(self, feat):
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat

class FeatureTrans(nn.Module):
    def __init__(self, input_dim=1024, out_dim=512, hidden=512):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(hidden, out_dim, kernel_size=1)
        )

    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.

    """
    def __init__(self, cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.DBA = []
        self.bestval = 0
        if args.loss == 'ce':  # crossentropy loss
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        image_encoder.train()
        lidar_encoder.train()
        radar_encoder.train()
        running_acc = 0.0
        gt_beam_all = []
        pred_beam_all = []
        # Train loop
        bar_format = '{l_bar}{bar:5}{r_bar}{bar:5b}'
        pbar = tqdm(dataloader_train, bar_format=bar_format)
        for data in pbar:

            # efficiently zero gradients
            optimizer.zero_grad(set_to_none=True)
            # create batch and move to GPU
            images = []
            lidars = []
            radars = []
            label = data['beam'][0]
            label = label.to(args.device)

            for i in range(config.seq_len):
                images.append(data['fronts'][i].to(args.device, dtype=torch.float32))
                lidars.append(data['lidars'][i].to(args.device, dtype=torch.float32))
                radars.append(data['radars'][i].to(args.device, dtype=torch.float32))
            bz, _, h, w = images[0].shape
            img_channel = images[0].shape[1]
            lidar_channel = lidars[0].shape[1]
            radar_channel = radars[0].shape[1]

            image_tensor = torch.stack(images, dim=1).view(bz * config.seq_len, img_channel, h, w)  # (bz*seq_len, img_c, h, w)
            lidar_tensor = torch.stack(lidars, dim=1).view(bz * config.seq_len, lidar_channel, h, w)  # (bz*seq_len, lidar_c, h, w)
            radar_tensor = torch.stack(radars, dim=1).view(bz * config.seq_len, radar_channel, h, w)  # (bz*seq_len, radar_c, h, w)

            with torch.no_grad():
                # layer1
                # image_feat_l1 = image_encoder.features.conv1(image_tensor)
                # image_feat_l1 = image_encoder.features.bn1(image_feat_l1)
                # image_feat_l1 = image_encoder.features.relu(image_feat_l1)
                # image_feat_l1 = image_encoder.features.maxpool(image_feat_l1)  # (bz*seq_len, 64, 64, 64)
                image_feat_l1 = image_encoder(image_tensor)

                # lidar_feat_l1 = lidar_encoder._model.conv1(lidar_tensor)
                # lidar_feat_l1 = lidar_encoder._model.bn1(lidar_feat_l1)
                # lidar_feat_l1 = lidar_encoder._model.relu(lidar_feat_l1)
                # lidar_feat_l1 = lidar_encoder._model.maxpool(lidar_feat_l1)  # (bz*seq_len, 64, 64, 64)
                lidar_feat_l1 = lidar_encoder(lidar_tensor)

                # radar_feat_l1 = radar_encoder._model.conv1(radar_tensor)
                # radar_feat_l1 = radar_encoder._model.bn1(radar_feat_l1)
                # radar_feat_l1 = radar_encoder._model.relu(radar_feat_l1)
                # radar_feat_l1 = radar_encoder._model.maxpool(radar_feat_l1)  # (bz*seq_len, 64, 64, 64)
                radar_feat_l1 = radar_encoder(radar_tensor)

                # image_feat_l1 = image_encoder.features.layer1(image_feat_l1)  # (bz*seq_len, 64, 64, 64)
                # lidar_feat_l1 = lidar_encoder._model.layer1(lidar_feat_l1)  # (bz*seq_len, 64, 64, 64)
                # radar_feat_l1 = radar_encoder._model.layer1(radar_feat_l1)  # (bz*seq_len, 64, 64, 64)

            l1_channel = image_feat_l1.shape[1]
            image_feat_l1 = image_feat_l1.view(bz * config.seq_len, l1_channel, -1)
            lidar_feat_l1 = lidar_feat_l1.view(bz * config.seq_len, l1_channel, -1)
            radar_feat_l1 = radar_feat_l1.view(bz * config.seq_len, l1_channel, -1)
            image_proj_l1 = image_projection_l1(image_feat_l1)  # (bz*seq_len, 64*64, 128)
            lidar_proj_l1 = lidar_projection_l1(lidar_feat_l1)  # (bz*seq_len, 64*64, 128)
            radar_proj_l1 = radar_projection_l1(radar_feat_l1)  # (bz*seq_len, 64*64, 128)

            split_num = int(image_proj_l1.shape[1] / 2)
            image_shared_l1 = image_proj_l1[:, :split_num, :]
            lidar_shared_l1 = lidar_proj_l1[:, :split_num, :]
            radar_shared_l1 = radar_proj_l1[:, :split_num, :]
            image_specific_l1 = image_proj_l1[:, split_num:, :]
            lidar_specific_l1 = lidar_proj_l1[:, split_num:, :]
            radar_specific_l1 = radar_proj_l1[:, split_num:, :]

            # Unsupervised Contrastive Learning for Shared Feature
            loss_contrast = criterion_contrast(image_shared_l1, lidar_shared_l1) + \
                            criterion_contrast(image_shared_l1, radar_shared_l1) + \
                            criterion_contrast(lidar_shared_l1, radar_shared_l1)
            loss_contrast = loss_contrast / 3.0

            # Feature Splitting with Distance for Specific Feature
            loss_distance = 0
            loss_distance = loss_distance - F.mse_loss(image_specific_l1, lidar_specific_l1)
            loss_distance = loss_distance - F.mse_loss(image_specific_l1, radar_specific_l1)
            loss_distance = loss_distance - F.mse_loss(lidar_specific_l1, radar_specific_l1)
            loss_distance = loss_distance / 3.0

            # Cross-modal Translation
            source_feat = []
            for m in args.source_domain:
                if m == 'image':
                    source_feat.append(image_shared_l1)
                elif m == 'lidar':
                    source_feat.append(lidar_shared_l1)
                elif m == 'radar':
                    source_feat.append(radar_shared_l1)
            source_feat_l1 = torch.cat(source_feat, dim=1)    # (bz*seq_len, 128, 64*64)
            if 'image' in args.source_domain:
                target_feat = image_feat_l1
            elif 'lidar' in args.source_domain:
                target_feat = lidar_feat_l1
            elif 'radar' in args.source_domain:
                target_feat = radar_feat_l1
            s2t_feat = feat_trans_l1(source_feat_l1)   # (bz*seq_len, 64, 64*64)
            loss_trans = torch.mean(torch.norm(s2t_feat-target_feat/torch.norm(target_feat, dim=1, keepdim=True), dim=1))
            # print(loss_trans)

            # loss backward
            loss_total = args.alpha_trans*loss_trans + \
                         args.alpha_contrast*loss_contrast + args.alpha_distance*loss_distance
            loss_total.backward()
            loss_epoch += float(loss_total.item())
            pbar.set_description(f'{loss_total.item():.2f}')
            pbar.set_postfix(t=f'{loss_trans.item():.3f}', c=f'{loss_contrast.item():.3f}',
                             d=f'{loss_distance.item():.3f}')
            num_batches += 1
            optimizer.step()

            self.cur_iter += 1
            writer.add_scalar('curr_iter_loss_trans', float(loss_trans.item()), self.cur_iter)
            writer.add_scalar('curr_iter_loss_contrast', float(loss_contrast.item()), self.cur_iter)
            writer.add_scalar('curr_iter_loss_distance', float(loss_distance.item()), self.cur_iter)

        # time.sleep(100)
        # pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
        # gt_beam_all = np.squeeze(np.concatenate(gt_beam_all, 0))
        # curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1, 2, 3])
        # DBA = compute_DBA_score(pred_beam_all, gt_beam_all, max_k=3, delta=5)
        # print('Train top beam acc: ', curr_acc, ' DBA score: ', DBA)
        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)

        self.cur_epoch += 1
        # writer.add_scalar('DBA_score_train', DBA, self.cur_epoch)
        # for i in range(len(curr_acc)):
        #     writer.add_scalars('curr_acc_train', {'beam' + str(i): curr_acc[i]}, self.cur_epoch)
        writer.add_scalar('curr_loss_train', loss_epoch, self.cur_epoch)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--id', type=str, default=time_id, help='Unique experiment identifier.')
    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--data_root', type=str, default='../Dataset', help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=15, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')	# default=24
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')	# /ibex/scratch/tiany0c/log
    parser.add_argument('--add_velocity', type = int, default=1, help='concatenate velocity map with angle map')
    parser.add_argument('--add_mask', type=int, default=0, help='add mask to the camera data')
    parser.add_argument('--enhanced', type=int, default=1, help='use enhanced camera data')
    parser.add_argument('--filtered', type=int, default=0, help='use filtered lidar data')
    parser.add_argument('--angle_norm', type=int, default=1, help='normlize the gps loc with unit, angle can be obtained')
    parser.add_argument('--custom_FoV_lidar', type=int, default=1, help='Custom FoV of lidar')
    parser.add_argument('--add_seg', type=int, default=0, help='add segmentation on 31&32 images')
    parser.add_argument('--loss', type=str, default='focal', help='crossentropy or focal loss')
    parser.add_argument('--scheduler', type=int, default=1, help='use scheduler to control the learning rate')
    parser.add_argument('--load_previous_best', type=int, default=0, help='load previous best pretrained model ')
    parser.add_argument('--temp_coef', type=int, default=1, help='apply temperature coefficience on the target')
    parser.add_argument('--Val', type=int, default=0, help='Val')
    parser.add_argument('--modality_missing', type=str, default=None, help='modality missing')
    parser.add_argument('--load_model_path', type=str, default=None, help='load model param for valuating')
    parser.add_argument('--temp', type=float, default=0.1, help='temp')
    parser.add_argument('--alpha_pred', type=float, default=0.5, help='alpha_pred')
    parser.add_argument('--alpha_trans', type=float, default=1.0, help='alpha_trans')
    parser.add_argument('--alpha_contrast', type=float, default=1.0, help='alpha_contrast')
    parser.add_argument('--alpha_distance', type=float, default=1.0, help='alpha_diatance')
    parser.add_argument('--encoder_weight', type=bool, default=True, help='load the pretrained weight for encoder')

    args = parser.parse_args()
    if args.logdir == 'log':
        args.logdir = os.path.join(args.logdir, args.id)
        source = '_'.join(args.source_domain)
        target = '_'.join(args.target_domain)
        tag = source + '2' + target
        args.logdir = args.logdir + '_' + tag
    if args.Val:
        args.logdir = args.logdir + '_val'
    if args.modality_missing is not None:
        args.logdir = args.logdir + '_' + args.modality_missing

    writer = SummaryWriter(log_dir=args.logdir)

    # Config
    config = GlobalConfig()
    config.add_velocity = args.add_velocity
    config.add_mask = args.add_mask
    config.enhanced = args.enhanced
    config.angle_norm = args.angle_norm
    config.filtered = args.filtered
    config.custom_FoV_lidar = args.custom_FoV_lidar
    config.add_seg = args.add_seg
    config.modality_missing = args.modality_missing
    config.data_root = args.data_root
    data_root = config.data_root  # path to the dataset

    # Set seed
    seed = 100
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch+CPU
    torch.cuda.manual_seed(seed)  # torch+GPU
    torch.use_deterministic_algorithms(False)
    g = torch.Generator()
    g.manual_seed(seed)

    # Dataloader
    trainval_root = data_root + '/Multi_Modal/'
    train_root_csv = 'ml_challenge_dev_multi_modal.csv'
    val_root = data_root + '/Adaptation_dataset_multi_modal/'
    val_root_csv = 'ml_challenge_data_adaptation_multi_modal.csv'

    development_set = CARLA_Data(root=trainval_root, root_csv=train_root_csv, config=config,
                                 test=False)  # development dataset 11k samples
    adaptation_set = CARLA_Data(root=val_root, root_csv=val_root_csv, config=config,
                                test=False)  # adaptation dataset 100 samples
    train_set = ConcatDataset([development_set, adaptation_set])
    train_size = int(0.9 * len(train_set))
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, len(train_set) - train_size])
    print('train_set:', len(train_set), 'val_set:', len(val_set))

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    # Modal
    image_encoder = ImageEncoder(weight=args.encoder_weight).to(args.device)
    lidar_encoder = LidarEncoder(weight=args.encoder_weight).to(args.device)
    radar_encoder = RadarEncoder(weight=args.encoder_weight).to(args.device)

    image_projection_l1 = ProjectHead(input_dim=64, hidden_dim=64, out_dim=128).to(args.device)
    lidar_projection_l1 = ProjectHead(input_dim=64, hidden_dim=64, out_dim=128).to(args.device)
    radar_projection_l1 = ProjectHead(input_dim=64, hidden_dim=64, out_dim=128).to(args.device)

    feat_trans_l1 = FeatureTrans(input_dim=128, hidden=128, out_dim=64).to(args.device)

    image_encoder = torch.nn.DataParallel(image_encoder)
    lidar_encoder = torch.nn.DataParallel(lidar_encoder)
    radar_encoder = torch.nn.DataParallel(radar_encoder)
    image_proj_l1 = torch.nn.DataParallel(image_projection_l1)
    lidar_proj_l1 = torch.nn.DataParallel(lidar_projection_l1)
    radar_proj_l1 = torch.nn.DataParallel(radar_projection_l1)
    feat_trans_l1 = torch.nn.DataParallel(feat_trans_l1)

    criterion_contrast = ContrastiveLoss()
    criterion_contrast = criterion_contrast.to(args.device)

    params = list(image_encoder.parameters()) + list(lidar_encoder.parameters()) + list(radar_encoder.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    if args.scheduler:  # Cyclic Cosine Decay Learning Rate
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=15,
                                        min_decay_lr=2.5e-6,
                                        restart_interval=10,
                                        restart_lr=12.5e-5,
                                        warmup_epochs=10,
                                        warmup_start_lr=2.5e-6)
    trainer = Engine()

    # Log args
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.Val:
        trainer.validate()
        print('Val finish')
    else:
        for epoch in range(trainer.cur_epoch, args.epochs):
            print('epoch:', epoch)
            trainer.train()