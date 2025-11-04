import torch
pth_file = 'E:/code/mmganguard/checkpoints/resnet_comparison/resnet18/latest_model.pth'  # 替换为你的 PTH 文件路径
model = torch.load(pth_file)

print(model)