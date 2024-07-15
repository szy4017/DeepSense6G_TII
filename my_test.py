import torch

model_path = '/data/szy4017/code/DeepSense6G_TII/experiments/FFM-csbimamba_TFM-attenmamba_base/best_model.pth'
model_dict = torch.load(model_path, map_location='cpu')

image_encoder_dict = {}
lidar_encoder_dict = {}
radar_encoder_dict = {}
fusion_model_dict = {}
for k in model_dict.keys():
    new_k = k.replace('module.', '')
    fusion_model_dict[new_k] = model_dict[k]

    if 'image_encoder' in k:
        new_k = k.replace('module.encoder.image_encoder.', '')
        image_encoder_dict[new_k] = model_dict[k]
    elif 'lidar_encoder' in k:
        new_k = k.replace('module.encoder.lidar_encoder.', '')
        lidar_encoder_dict[new_k] = model_dict[k]
    elif 'radar_encoder' in k:
        new_k = k.replace('module.encoder.radar_encoder.', '')
        radar_encoder_dict[new_k] = model_dict[k]

print(fusion_model_dict.keys())
print(image_encoder_dict.keys())
print(lidar_encoder_dict.keys())
print(radar_encoder_dict.keys())

torch.save(fusion_model_dict, 'modality_rebuild/fusion_model.pth')
torch.save(image_encoder_dict, 'modality_rebuild/image_encoder.pth')
torch.save(lidar_encoder_dict, 'modality_rebuild/lidar_encoder.pth')
torch.save(radar_encoder_dict, 'modality_rebuild/radar_encoder.pth')
print('Done!')


# model_path = 'modality_rebuild/mambafusion.pth'
# model_dict = torch.load(model_path, map_location='cpu')
# new_state_dict = {}
# for k in model_dict.keys():
#     new_k = k.replace('module.', '')
#     new_state_dict[new_k] = model_dict[k]
#
# print(new_state_dict.keys())
#
# torch.save(new_state_dict, 'modality_rebuild/mamba_fusion.pth')
