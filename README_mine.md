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
* train_time: 14min\*150=2100min=35h

# 实验中的问题
## Problem 1
使用mamba替换transformer进行20to5的多步预测是，模型训练在前期（第4个epoch）就出现梯度爆炸（loss=NaN）。

## Solution (1)
使用梯度裁剪，并减小学习率
* torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
* lr: 5e-4 -> 1e-4
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
结果：训练到第7个epoch依然出现梯度爆炸（loss=NaN）。

## Solution (2)
设置更短的history(10to5)，也匹配baseline的设定
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
结果：训练到第6个epoch依然出现梯度爆炸（loss=NaN）。

## Solution (3)
在mambafusion中采用双边mamba编码
```
python train2_seq_30to5.py --epochs 50 --batch_size 12 --lr 1e-4
```
* self.ln1 = nn.LayerNorm(ln_size)
* x_fused = torch.add(torch.mul(x_bm, x_relu), torch.mul(x_fm, x_bm))
* torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
结果：解决梯度爆炸问题，loss正常下降且性能稳定，best_DBA_score_scenario_all=0.9142 > 0.4671 (baseline)

### Commands
tensorboard --logdir log --host=10.15.198.46 --port=6007
python train2_seq.py --epochs 30 --batch_size 24 --logdir 'log/20240619_165326'

