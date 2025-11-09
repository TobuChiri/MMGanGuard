import torch
pth_file = 'E:/code/mmganguard/checkpoints/final_co_occurrence_model.pth'  # 替换为你的 PTH 文件路径
model = torch.load(pth_file)

print(model)