import torch

# model_path = 'log/log_mambafusion_50epoch_5to1_bimamba/best_model.pth'
# model_dict = torch.load(model_path, map_location='cpu')
#
# image_encoder_dict = {}
# lidar_encoder_dict = {}
# radar_encoder_dict = {}
# for k in model_dict.keys():
#     if 'image_encoder' in k:
#         new_k = k.replace('module.encoder.image_encoder.', '')
#         image_encoder_dict[new_k] = model_dict[k]
#     elif 'lidar_encoder' in k:
#         new_k = k.replace('module.encoder.lidar_encoder.', '')
#         lidar_encoder_dict[new_k] = model_dict[k]
#     elif 'radar_encoder' in k:
#         new_k = k.replace('module.encoder.radar_encoder.', '')
#         radar_encoder_dict[new_k] = model_dict[k]
#
# print(image_encoder_dict.keys())
# print(lidar_encoder_dict.keys())
# print(radar_encoder_dict.keys())
#
# torch.save(image_encoder_dict, 'modality_rebuild/image_encoder.pth')
# torch.save(lidar_encoder_dict, 'modality_rebuild/lidar_encoder.pth')
# torch.save(radar_encoder_dict, 'modality_rebuild/radar_encoder.pth')
# print('Done!')


model_path = 'modality_rebuild/mambafusion.pth'
model_dict = torch.load(model_path, map_location='cpu')
new_state_dict = {}
for k in model_dict.keys():
    new_k = k.replace('module.', '')
    new_state_dict[new_k] = model_dict[k]

print(new_state_dict.keys())

torch.save(new_state_dict, 'modality_rebuild/mamba_fusion.pth')
