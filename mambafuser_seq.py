import math
from collections import deque
import random
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from mamba_ssm import Mamba


class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(weights=True)
        self.features.fc = nn.Sequential()
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

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


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18(weights=True)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        # for param in self._model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature
        return features

class MambaBlock(nn.Module):
    """ a bi-branch Mamba block """

    def __init__(self, n_embd, ln_size, d_state, d_conv, expand):
        super().__init__()
        self.ln1 = nn.LayerNorm(ln_size)
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.forward_mamba = Mamba(d_model=n_embd,
                                   d_state=d_state,
                                   d_conv=d_conv,
                                   expand=expand)
        self.backward_mamba = Mamba(d_model=n_embd,
                                    d_state=d_state,
                                    d_conv=d_conv,
                                    expand=expand)

    def forward(self, x):
        B, T, C = x.size()
        x_ln = self.ln1(x)

        x_fc1 = self.fc1(x_ln)
        # forward mamba
        x_fm = self.forward_mamba(x_fc1)
        # backward mamba
        x_fc1 = torch.flip(x_fc1, dims=[1])
        x_bm = self.backward_mamba(x_fc1)

        x_fc2 = self.fc2(x_fc1)
        x_relu = self.relu(x_fc2)

        # fuse forward and backward feature
        x_fused = torch.add(torch.mul(x_bm, x_relu), torch.mul(x_fm, x_bm))

        return x_fused

class MambaFusion(nn.Module):
    """  the full mamba model, with a context size of block_size """

    def __init__(self, n_embd, ln_size, d_state, d_conv, expand, n_layer,
                 vert_anchors, horz_anchors, seq_len, embd_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (self.config.n_views + 2) * seq_len * vert_anchors * horz_anchors + 2, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.mambablocks = nn.Sequential(*[MambaBlock(n_embd, ln_size, d_state, d_conv, expand)
                                         for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, radar_tensor, gps):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            gps (tensor): ego-gps
        """

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        #         print('transfo',self.config.n_views , self.seq_len)

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)
        radar_tensor = radar_tensor.view(bz, self.seq_len, -1, h, w)

        # channel swapping for image_tensor, lidar tensor and radar tensor
        s1 = self.n_embd // 3
        s2 = self.n_embd // 3 * 2
        cs_image_tensor = torch.cat((image_tensor[:, :, :s1, ...], lidar_tensor[:, :, s1:s2, ...], radar_tensor[:, :, s2:, ...]), dim=2)
        cs_lidar_tensor = torch.cat((lidar_tensor[:, :, :s1, ...], radar_tensor[:, :, s1:s2, ...], image_tensor[:, :, s2:, ...]), dim=2)
        cs_radar_tensor = torch.cat((radar_tensor[:, :, :s1, ...], image_tensor[:, :, s1:s2, ...], lidar_tensor[:, :, s2:, ...]), dim=2)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat(
            [cs_image_tensor, cs_lidar_tensor, cs_radar_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        # print(token_embeddings.shape)
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)
        token_embeddings = torch.cat([token_embeddings, gps], dim=1)
        # add (learnable) positional embedding and gps embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings)  # (B, an * T, C)

        # bi-path mamba encoding
        x = self.mambablocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        pos_tensor_out = x[:, (self.config.n_views + 2) * self.seq_len * self.vert_anchors * self.horz_anchors:, :]
        x = x[:, :(self.config.n_views + 2) * self.seq_len * self.vert_anchors * self.horz_anchors, :]

        x = x.view(bz, (self.config.n_views + 2) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views * self.seq_len, :, :, :].contiguous().view(
            bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views * self.seq_len:(self.config.n_views + 1) * self.seq_len, :, :,
                           :].contiguous().view(bz * self.seq_len, -1, h, w)
        radar_tensor_out = x[:, (self.config.n_views + 1) * self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len,
                                                                                                      -1, h, w)
        return image_tensor_out, lidar_tensor_out, radar_tensor_out, pos_tensor_out

class TimeMamba(nn.Module):
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.maxpool = nn.MaxPool1d(kernel_size=512)
        self.avgpool = nn.AvgPool1d(kernel_size=512)
        self.mlp = nn.Sequential(
            nn.Linear(5, 5),
            nn.Softmax(dim=-1),
        )
        self.mlp_gps = nn.Sequential(
            nn.Linear(2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, image_features, lidar_features, radar_features, gps_features):
        """
        image_features (tensor): (bz, seq_len, 512)
        lidar_features (tensor): (bz, seq_len, 512)
        radar_features (tensor): (bz, seq_len, 512)
        gps_features (tensor): (bz, 2, 512)

        return: fused_features (tensor): (bz, 512)
        """
        image_features = self.mamba(image_features)
        lidar_features = self.mamba(lidar_features)
        radar_features = self.mamba(radar_features)

        image_atten = self.maxpool(image_features) + self.avgpool(image_features)
        image_atten = self.mlp(image_atten.squeeze(-1)).unsqueeze(-1).expand(-1, -1, 512)
        image_feat = (image_features * image_atten).sum(dim=1, keepdim=True)

        lidar_atten = self.maxpool(lidar_features) + self.avgpool(lidar_features)
        lidar_atten = self.mlp(lidar_atten.squeeze(-1)).unsqueeze(-1).expand(-1, -1, 512)
        lidar_feat = (lidar_features * lidar_atten).sum(dim=1, keepdim=True)

        radar_atten = self.maxpool(radar_features) + self.avgpool(radar_features)
        radar_atten = self.mlp(radar_atten.squeeze(-1)).unsqueeze(-1).expand(-1, -1, 512)
        radar_feat = (radar_features * radar_atten).sum(dim=1, keepdim=True)

        gps_atten = self.maxpool(gps_features) + self.avgpool(gps_features)
        gps_atten = self.mlp_gps(gps_atten.squeeze(-1)).unsqueeze(-1).expand(-1, -1, 512)
        gps_feat = (gps_features * gps_atten).sum(dim=1, keepdim=True)

        fused_features = torch.cat([image_feat, lidar_feat, radar_feat, gps_feat], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features

class EncoderWithMamba(nn.Module):
    """
    Multi-scale Fusion Mamba for image + Radar + LiDAR feature fusion
    Use bibranch Mamba block for feature fusion
    Use Mamba block for time fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        self.image_encoder = ImageCNN(512, normalize=True)
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)
        if config.add_velocity:
            self.radar_encoder = LidarEncoder(num_classes=512, in_channels=2)
        else:
            self.radar_encoder = LidarEncoder(num_classes=512, in_channels=1)
        self.missing = self.config.modality_missing

        self.vel_emb1 = nn.Linear(2, 64)
        self.vel_emb2 = nn.Linear(64, 128)
        self.vel_emb3 = nn.Linear(128, 256)
        self.vel_emb4 = nn.Linear(256, 512)

        self.mambafusion1 = MambaFusion(n_embd=64,
                                        ln_size=(962, 64),
                                        d_state=16,
                                        d_conv=4,
                                        expand=2,
                                        n_layer=config.n_layer,
                                        vert_anchors=config.vert_anchors,
                                        horz_anchors=config.horz_anchors,
                                        seq_len=config.seq_len,
                                        embd_pdrop=config.embd_pdrop,
                                        config=config)
        self.mambafusion2 = MambaFusion(n_embd=128,
                                        ln_size=(962, 128),
                                        d_state=16,
                                        d_conv=4,
                                        expand=2,
                                        n_layer=config.n_layer,
                                        vert_anchors=config.vert_anchors,
                                        horz_anchors=config.horz_anchors,
                                        seq_len=config.seq_len,
                                        embd_pdrop=config.embd_pdrop,
                                        config=config)
        self.mambafusion3 = MambaFusion(n_embd=256,
                                        ln_size=(962, 256),
                                        d_state=16,
                                        d_conv=4,
                                        expand=2,
                                        n_layer=config.n_layer,
                                        vert_anchors=config.vert_anchors,
                                        horz_anchors=config.horz_anchors,
                                        seq_len=config.seq_len,
                                        embd_pdrop=config.embd_pdrop,
                                        config=config)
        self.mambafusion4 = MambaFusion(n_embd=512,
                                        ln_size=(962, 512),
                                        d_state=16,
                                        d_conv=4,
                                        expand=2,
                                        n_layer=config.n_layer,
                                        vert_anchors=config.vert_anchors,
                                        horz_anchors=config.horz_anchors,
                                        seq_len=config.seq_len,
                                        embd_pdrop=config.embd_pdrop,
                                        config=config)

        self.time_mamba = TimeMamba(d_model=512,
                                    d_state=16,
                                    d_conv=4,
                                    expand=2)

    def modality_missing(self, input_tensor, miss):
        image, lidar, radar = input_tensor
        device = image.device
        if miss == None:
            return [image, lidar, radar]
            # lidar_missing = torch.zeros_like(lidar).to(device)
            # radar_missing = torch.zeros_like(radar).to(device)
            # return [image, lidar_missing, radar_missing]
        elif miss == 'image':
            image_missing = torch.zeros_like(image).to(device)
            return [image_missing, lidar, radar]
        elif miss == 'lidar':
            lidar_missing = torch.zeros_like(lidar).to(device)
            return [image, lidar_missing, radar]
        elif miss == 'radar':
            radar_missing = torch.zeros_like(radar).to(device)
            return [image, lidar, radar_missing]
        elif miss == 'lidar_radar' or miss == 'radar_lidar':
            lidar_missing = torch.zeros_like(lidar).to(device)
            radar_missing = torch.zeros_like(radar).to(device)
            return [image, lidar_missing, radar_missing]

    def forward(self, image_list, lidar_list, radar_list, gps, rebuild_modality_feat_list=None):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            gps (tensor): input gps
        '''
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1]
        lidar_channel = lidar_list[0].shape[1]
        radar_channel = radar_list[0].shape[1]

        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel,
                                                           h, w)  # (bz*seq_len, img_c, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel,
                                                           h, w)  # (bz*seq_len, lidar_c, h, w)
        radar_tensor = torch.stack(radar_list, dim=1).view(bz * self.config.seq_len, radar_channel,
                                                           h, w)  # (bz*seq_len, radar_c, h, w)

        # input tensor missing
        if self.missing is not None:
            image_tensor, lidar_tensor, radar_tensor = self.modality_missing([image_tensor, lidar_tensor, radar_tensor], self.missing)

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)  # (bz*seq_len, 64, 64, 64)

        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.relu(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)  # (bz*seq_len, 64, 64, 64)

        radar_features = self.radar_encoder._model.conv1(radar_tensor)
        radar_features = self.radar_encoder._model.bn1(radar_features)
        radar_features = self.radar_encoder._model.relu(radar_features)
        radar_features = self.radar_encoder._model.maxpool(radar_features)  # (bz*seq_len, 64, 64, 64)

        image_features = self.image_encoder.features.layer1(image_features)  # (bz*seq_len, 64, 64, 64)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)  # (bz*seq_len, 64, 64, 64)
        radar_features = self.radar_encoder._model.layer1(radar_features)  # (bz*seq_len, 64, 64, 64)

        # fusion at (B, 64, 64, 64)
        if rebuild_modality_feat_list is not None and self.missing is not None:
            if self.missing == 'image':
                if self.training:
                    image_features_rebuild = rebuild_modality_feat_list[0]
                    image_features_rebuild = image_features_rebuild.view(bz * self.config.seq_len, 64, 64, 64)
                    if random.choice([True, False, False, False]):
                        image_features = image_features_rebuild
                else:
                    image_features = rebuild_modality_feat_list[0]
                    image_features = image_features.view(bz * self.config.seq_len, 64, 64, 64)
            elif self.missing == 'lidar':
                lidar_features = rebuild_modality_feat_list[0]
                lidar_features = lidar_features.view(bz * self.config.seq_len, 64, 64, 64)
            elif self.missing == 'radar':
                radar_features = rebuild_modality_feat_list[0]
                radar_features = radar_features.view(bz * self.config.seq_len, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        radar_embd_layer1 = self.avgpool(radar_features)
        gps_embd_layer1 = self.vel_emb1(gps)

        image_features_layer1, lidar_features_layer1, radar_features_layer1, gps_features_layer1 = self.mambafusion1(
            image_embd_layer1, lidar_embd_layer1, radar_embd_layer1, gps_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear')
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear')
        radar_features_layer1 = F.interpolate(radar_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        radar_features = radar_features + radar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        radar_features = self.radar_encoder._model.layer2(radar_features)

        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        radar_embd_layer2 = self.avgpool(radar_features)
        gps_embd_layer2 = self.vel_emb2(gps_features_layer1)

        image_features_layer2, lidar_features_layer2, radar_features_layer2, gps_features_layer2 = self.mambafusion2(
            image_embd_layer2, lidar_embd_layer2, radar_embd_layer2, gps_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear')
        radar_features_layer2 = F.interpolate(radar_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2
        radar_features = radar_features + radar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        radar_features = self.radar_encoder._model.layer3(radar_features)

        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        radar_embd_layer3 = self.avgpool(radar_features)
        gps_embd_layer3 = self.vel_emb3(gps_features_layer2)

        image_features_layer3, lidar_features_layer3, radar_features_layer3, gps_features_layer3 = self.mambafusion3(
            image_embd_layer3, lidar_embd_layer3, radar_embd_layer3, gps_embd_layer3)

        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear')
        radar_features_layer3 = F.interpolate(radar_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3
        radar_features = radar_features + radar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        radar_features = self.radar_encoder._model.layer4(radar_features)

        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)  # (bz*seq_len, 512, 8, 8)
        lidar_embd_layer4 = self.avgpool(lidar_features)  # (bz*seq_len, 512, 8, 8)
        radar_embd_layer4 = self.avgpool(radar_features)  # (bz*seq_len, 512, 8, 8)
        gps_embd_layer4 = self.vel_emb4(gps_features_layer3)  # (bz, 2, 512)

        image_features_layer4, lidar_features_layer4, radar_features_layer4, gps_features_layer4 = self.mambafusion4(
            image_embd_layer4, lidar_embd_layer4, radar_embd_layer4, gps_embd_layer4)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        radar_features = radar_features + radar_features_layer4

        image_features = self.image_encoder.features.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)  # (bz, seq_len, 512)
        lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.config.seq_len, -1)
        radar_features = self.radar_encoder._model.avgpool(radar_features)  # (bz, seq_len, 512)
        radar_features = torch.flatten(radar_features, 1)
        radar_features = radar_features.view(bz, self.config.seq_len, -1)  # (bz, seq_len, 512)
        gps_features = gps_features_layer4  # (bz, 2, 512)

        # time fusion with mamba
        if self.config.TFM:
            # multi-modal feature shape: (bz, seq_len, 512)
            fused_features = self.time_mamba(image_features, lidar_features, radar_features, gps_features)  # (bz, 512)
            return fused_features

        fused_features = torch.cat([image_features, lidar_features, radar_features, gps_features],
                                   dim=1)  # (bz, 17, 512)
        # fused_features = torch.cat([image_features, lidar_features, radar_features], dim=1)

        fused_features = torch.sum(fused_features, dim=1)  # (bz, 512)

        return fused_features


class MambaFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device, pretrain_weight=False):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        # self.encoder = Encoder(config).to(self.device)
        self.encoder = EncoderWithMamba(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                        ).to(self.device)
        # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        # self.output = nn.Linear(64, 2).to(self.device)
        if pretrain_weight:
            self.load_pretrained_weight()

    def load_pretrained_weight(self):
        model_dict = torch.load('fusion_model.pth')
        self.load_state_dict(model_dict)
        print('Pretrained weights loaded from mamba_fusion.pth')
        
    def forward(self, image_list, lidar_list, radar_list, gps, rebuild_modality_feat_list=None):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            gps (tensor): input gps
        '''
        fused_features = self.encoder(image_list, lidar_list, radar_list, gps, rebuild_modality_feat_list)
        z = self.join(fused_features)

       

        return z
