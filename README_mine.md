# DeepSense6G_TII的复现
## Step 1
配置虚拟环境：
* env_name: `deepsense`
* env_path: `/data/szy4017/miniconda3/envs/deepsense`
* dataset_path = `/data/share/dataset/DeepSense6G`

## Step 2
模型训练
* checkpoint for best_model.pth: 原作者给了下载链接，但是没法下载，已经在GitHub上提了issue
* epoch: 150
* batch: 2392, 1.3s for one batch
* batch_size: 6
* GPU memory: 4.5+3.8+3.8+3.8+3.8+3.8=23.5G for 6 GPUs
* train_time: 1.2\*2392\*150=430560s=120h for 6 GPUs

retrain base model
* batch_size: 24
* GPU memory: 10*6=60G for 6 GPUs
* train_time: 8min\*150=1200min=20h

train mamba modal
* batch_size: 24
* GPU memory: 5\*6=30G for 6 GPUs
* train_time: 14min\*30=420min=7h

train bimamba modal
```
python train2_seq.py --epochs 50 --batch_size 24
```
* batch_size: 24
* GPU memory: 7\*6=42G for 6 GPUs
* Params: 103MB
* train_time: 14min\*50=420=7h
* dataset split: trainval=adaset+devset, train:val=9:1

## 实验中的问题
### Problem 1
使用mamba替换transformer进行20to5的多步预测是，模型训练在前期（第4个epoch）就出现梯度爆炸（loss=NaN）。

### Solution (1)
使用梯度裁剪，并减小学习率
* torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
* lr: 5e-4 -> 1e-4
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
结果：训练到第7个epoch依然出现梯度爆炸（loss=NaN）。

### Solution (2)
设置更短的history(10to5)，也匹配baseline的设定
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
结果：训练到第6个epoch依然出现梯度爆炸（loss=NaN）。

### Solution (3)
在mambafusion中采用双边mamba编码
* self.ln1 = nn.LayerNorm(ln_size)
* x_fused = torch.add(torch.mul(x_bm, x_relu), torch.mul(x_fm, x_bm))
* torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
结果：解决梯度爆炸问题，loss正常下降且性能稳定，best_DBA_score_scenario_all=0.9142 > 0.4671 (baseline)

# TODO
## missing modality rebuild
1. 测试baseline: upper limit->multi-training and inference; lower limit->multi-training and missing inference;
2. 基于SimMMDG构建modality rebuild模块，实现对missing modality的生成;
3. 主要参考指标 DBA_score_val_scenario_all
```
python train2_seq.py --batch_size 24 --Val 1 --modality_missing image --loda_model_path log/log_mambafusion_50epoch_5to1_bimamba/best_model.pth
```

| model | upper  | image-rebuild | lower-image-miss | radar-rebuild | lower-radar-miss | lidar-rebuild | lower-lidar-miss | radar-lidar-rebuild | lower-radar-lidar-miss |
|-------|--------|---------------|------------------|---------------|------------------|---------------|------------------|---------------------|------------------------|
| mamba | 0.9108 |               | 0.1903           |               | 0.9042           |               | 0.8929           |                     | 0.8864                 |
log_path: log/log_mambafusion_50epoch_5to1_bimamba/best_model.pth

Metrics on no missing
![log_no_missing](./Materials/log_no_missing.png)

Metrics on image missing
![log_image_missing](./Materials/log_image_missing.png)

Metrics on radar missing
![log_radar_missing](./Materials/log_radar_missing.png)

Metrics on lidar missing
![log_lidar_missing](./Materials/log_lidar_missing.png)

Metrics on radar and lidar missing
![log_lidar_missing](./Materials/log_radar_lidar_missing.png)

## modality rebuild module
1. 模态的特征提取采用原有结构: ImageEncoder, RadarEncoder, LidarEncoder;
2. FusionEncoder用于融合特征预测beam;
3. FeatureTrans用于将source modality向target modality转换;
4. loss_total = loss_pred + loss_trans + loss_contrast + loss_distance
```
export PYTHONPATH=$PYTHONPATH:/data/szy4017/code/DeepSense6G_TII
CUDA_VISIBLE_DEVICES=0,1,2 python train_image_radar_lidar_rebuild.py -s image radar -t lidar --lr 1e-3 --epochs 30 --batch_size 24
```

## 实验中的问题
### Problem 1
损失下降比较慢，30epoch loss_total (14.5->7.6)，预测性能表现也很差，0.3756（训练时多模态融合的预测性能），
在推理时应该用source+rebuild进行融合的预测性能。
```
python train_image_radar_lidar_rebuild.py -s image radar -t lidar --epochs 30 --batch_size 24
```

### Solution (1)
调整学习率
* lr: 1e-4 -> 1e-3 or 5e-4
```
python train_image_radar_lidar_rebuild.py -s image radar -t lidar --lr 5e-4 --epochs 30 --batch_size 24
```
结果：效果不明显，调大lr，loss开始震荡

### Solution (2)
采用mambafusion里面的encoder参数作为预训练参数
```
CUDA_VISIBLE_DEVICES=3,4,5 python train_image_radar_lidar_rebuild.py -s image radar -t lidar --epochs 30 --batch_size 24
```
结果：训练依然不收敛，DBA_score_train=0.2376<0.3756(baseline)，
观察loss变化：loss_pred=12->6; loss_trans=1.4->1.4; loss_contrast=6.1->5; loss_distance=0->-1.3
loss_pred的误差太大，loss_trans没有得到有效训练

### Solution (3)
用mambafusion的frozen参数直接进行融合的预测，不用loss_pred，重点训练FeatureTrans模块
```
python train_image_radar_lidar_rebuild.py -s image radar -t lidar --epochs 30 --batch_size 24 --lr 1e-3
```
结果：loss几乎不下降，无法有效训练，loss_contrast=12.0；loss_trans=1.0；loss_distance=0.0

### Solution (4)
调整多个loss的权重，对于对比学习的loss_contrast，需要解决正负样本不均衡的问题，用batch_size=2试一试，进一步需要把label引入到对比损失计算中
把源域改成lidar和radar，目标域改成image
* batch_size: 2
* alpha_contrast: 0.5
* alpha_trans: 5.0
* alpha_distance: 2.0
* s: lidar radar, t: image
```
python train_image_radar_lidar_rebuild.py -s lidar radar -t image --batch_size 2 --epoch 30 --alpha_contrast 0.5 --alpha_trans 5.0 --alpha_distance 2.0
```
结果：batch_size=2可以帮助损失下降加快，loss_contrast=8.8->1.1，loss_trans=1.0->0.5，loss_diatance=0.0(可能是前期变化不大)，
在validate中，预测性能依旧很差，DBA_score=0.19，并且越训练性能还降低了

### Solution (5)
将mambafusion的参数也加入训练，进行finetune，多了一个loss_fusion，并且屏蔽掉loss_contrast和loss_diatance，只用loss_fusion和loss_trans进行训练
* loss_total = 5.0 \* loss_trans + loss_fusion
```
python train_image_radar_lidar_rebuild.py -s lidar radar -t image --batch_size 8 --epoch 30 --alpha_contrast 0.5 --alpha_trans 5.0 --alpha_distance 10.0 --modality_missing image
```
结果：loss下降正常，DBA score可以达到0.79，但是fintune之后的mambafusion在正常的no missing任务中又崩掉了

Log on DBA score and loss
![log_lidar_missing](./Materials/log-finetune_mambafusion_lidar_radar2image.png)

### Solution (6)
调整mambafusion在finetune中的lr，采用rebuild数据和正常数据的混合训练，frozen掉编码器部分的参数
* lr(mambafusion): 1e-4 -> 1e-6
```
python train_image_radar_lidar_rebuild.py -s lidar radar -t image --batch_size 8 --epoch 30 --alpha_contrast 0.05 --alpha_trans 5.0 --alpha_distance 2.0 --modality_missing image
```

# Experiments

## Table I
MambaFusion vs other method, DBA score

I: Image; L: LiDAR; R: Radar; G: GPS

| Method    | Modality | Parameter | Overall | Scenario 31 | Scenario 32 | Scenario 33 | Scenario 34 |
|-----------|----------|-----------|---------|-------------|-------------|-------------|-------------|
| baseline1 |          |           |         |             |             |             |             |
| baseline2 |          |           |         |             |             |             |             |
| baseline3 |          |           |         |             |             |             |             |
| baseline4 |L+R+G     | 78422528  | 0.9005  | 1.0         | 0.8883      | 0.8775      | 0.8725      |
| baseline5 |          |           |         |             |             |             |             |
| ours      |I+G       | 78118308  | 0.8736  | 0.9994      | 0.8619      | 0.8587      | 0.8323      |
| ours      |I+L+G     | 92288548  | 0.8803  | 1.0         | 0.8669      | 0.8640      | 0.8462      |
| ours      |I+R+G     | 92291684  | 0.8754  | 0.9997      | 0.8623      | 0.8501      | 0.8487      |
| ours      |L+R+G     | 82177252  | 0.8157  | 1.0         | 0.8330      | 0.7806      | 0.7312      |
| ours      |I+L+R+G   |103461924  | 0.8918  | 1.0         | 0.8776      | 0.8759      | 0.8918      |
```
python train2_seq.py --logdir experiments/FFM_TFM_base --batch_size 24
python train2_seq.py --logdir experiments/FFM-csbimamba_TFM-attenmamba_base --batch_size 24
python train2_seq.py --logdir experiments/FFM-csbimamba_TFM-attenmamba_img_lidar --batch_size 24 --modality_missing radar
python train2_seq.py --logdir experiments/FFM-csbimamba_TFM-attenmamba_img_radar --batch_size 24 --modality_missing lidar
python train2_seq.py --logdir experiments/FFM-csbimamba_TFM-attenmamba_lidar_radar --batch_size 24 --modality_missing image
python train2_seq.py --logdir experiments/FFM-csbimamba_TFM-attenmamba_image --batch_size 24 --modality_missing lidar_radar
```

## Table II
Ablation study in MambaFusion module (FFM TFM), DBA score, training GPU memory, training GPU hours, inference speed

| FFM  | TFM | DBA score | GPU memory | GPU hours | FPS |
|------|-----|-----------|------------|-----------|-----|
|      |     |           |            |           |     |
| ✅    |     |           |            |           |     |
|      | ✅    |           |            |           |     |
| ✅     | ✅    |           |            |           |     |

* FFM: Feature Fusion Mamba
* TFM: Time Fusion Mamba
* baseline: Transfuser_TII -> TODO
* Line 3: Transformer_TII + TFM -> TODO

```
python train2_seq.py --logdir experiments/FFM-csbimamba_ablation --TFM 0 --batch_size 24
```

## Table III
Performance in modality missing
modality_missing_type: zerolike, randlike

| Method     | Image missing  | lidar missing  | radar missing  | lidar & radar missing |
|------------|----------------|----------------|----------------|-----------------------|
| baseline   |                |                |                |                       |
| ours       | 0.2695, 0.2793 | 0.6958, 0.6960 | 0.6961, 0.2386 | 0.6966, 0.2346        |
| ours + MRM | 0.5941, 0.5968 | 0.8012, 0.7855 | 0.8129, 0.6542 | 0.7523, 0.6863        |

* baseline: Transfuser_TII -> TODO
* MRM: Modality Rebuilding Module

```
# zerolike and randlike settings
python train2_seq.py --logdir experiments/mambafuser_image_missing --Val 1 --batch_size 24 --modality_missing image --load_model_path experiments/FFM-csbimamba_TFM-attenmamba_base/best_model.pth 
python train2_seq.py --logdir experiments/mambafuser_lidar_missing --Val 1 --batch_size 24 --modality_missing lidar --load_model_path experiments/FFM-csbimamba_TFM-attenmamba_base/best_model.pth 
python train2_seq.py --logdir experiments/mambafuser_radar_missing --Val 1 --batch_size 24 --modality_missing radar --load_model_path experiments/FFM-csbimamba_TFM-attenmamba_base/best_model.pth 
python train2_seq.py --logdir experiments/mambafuser_lidar_radar_missing --Val 1 --batch_size 24 --modality_missing lidar_radar --load_model_path experiments/FFM-csbimamba_TFM-attenmamba_base/best_model.pth 

python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_image_missing --batch_size 24 -s lidar radar -t image --modality_missing image --modality_missing_type zerolike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_image_missing --batch_size 24 -s lidar radar -t image --modality_missing image --modality_missing_type randlike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_lidar_missing --batch_size 24 -s image radar -t lidar --modality_missing lidar --modality_missing_type zerolike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_lidar_missing --batch_size 24 -s image radar -t lidar --modality_missing lidar --modality_missing_type randlike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_radar_missing --batch_size 24 -s image lidar -t radar --modality_missing radar --modality_missing_type zerolike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_radar_missing --batch_size 24 -s image lidar -t radar --modality_missing radar --modality_missing_type randlike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_lidar_radar_missing --batch_size 24 -s image -t lidar radar --modality_missing lidar_radar --modality_missing_type zerolike
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_lidar_radar_missing --batch_size 24 -s image -t lidar radar --modality_missing lidar_radar --modality_missing_type randlike
```

## Table IV
Ablation study in modality rebuilding

| loss_fusion | loss_trans | loss_contrast | loss_distance | L,R->I | I,R->L | I,L->R | I->L,R |
|-------------|------------|---------------|---------------|--------|--------|--------|--------|
| ✅           |            |               |               | 0.5048 |        |        |        |
| ✅           | ✅           |               |               | 0.5622 |        |        |        |
| ✅           | ✅           | ✅              |               | 0.5745 |        |        |        |
| ✅           | ✅           | ✅              | ✅              | 0.5941 |        |        |        |

增加一个使用不同loss组合的,loss下降图

```
# TODO (in randlike setting)
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_image_missing --batch_size 24 -s lidar radar -t image --modality_missing image --alpha_trans 0 --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_image_missing --batch_size 24 -s lidar radar -t image --modality_missing image --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_constrast_image_missing --batch_size 24 -s lidar radar -t image --modality_missing image --alpha_distance 0

python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_lidar_missing --batch_size 24 -s image radar -t lidar --modality_missing lidar --alpha_trans 0 --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_lidar_missing --batch_size 24 -s image radar -t lidar --modality_missing lidar --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_constrast_lidar_missing --batch_size 24 -s image radar -t lidar --modality_missing lidar --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_radar_missing --batch_size 24 -s image lidar -t radar --modality_missing radar --alpha_trans 0 --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_radar_missing --batch_size 24 -s image lidar -t radar --modality_missing radar --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_constrast_radar_missing --batch_size 24 -s image lidar -t radar --modality_missing radar --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_lidar_radar_missing --batch_size 24 -s image -t lidar radar --modality_missing lidar_radar --alpha_trans 0 --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_lidar_radar_missing --batch_size 24 -s image -t lidar radar --modality_missing lidar_radar --alpha_contrast 0 --alpha_distance 0
python train_mambafuser_modality_rebuild.py --logdir experiments/rebuilding_fusion_trans_constrast_lidar_radar_missing --batch_size 24 -s image -t lidar radar --modality_missing lidar_radar --alpha_distance 0
```

### Commands
```
export PYTHONPATH=$PYTHONPATH:/data/szy4017/code/DeepSense6G_TII
tensorboard --logdir log --host=10.15.198.46 --port=6008
tensorboard --logdir experiments --host=10.15.198.46 --port=6008
python train2_seq.py --epochs 30 --batch_size 24 --logdir 'log/20240619_165326'
```

