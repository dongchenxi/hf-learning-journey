import torch
from torch import nn


# 1. 创建一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = SimpleModel()


# 2. 查看原始模型参数
print("=== Before Freezing ===")

for name, param in model.named_parameters():
    print(
        f"{name}: "
        f"shape={tuple(param.shape)}, "
        f"requires_grad={param.requires_grad}"
    )


# 3. 冻结 layer1
for param in model.layer1.parameters():
    param.requires_grad = False


# 4. 查看冻结后的参数
print("\n=== After Freezing ===")

for name, param in model.named_parameters():
    print(
        f"{name}: "
        f"requires_grad={param.requires_grad}"
    )


# 5. 统计参数数量
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_params = total_params - trainable_params


print("\n=== Parameter Statistics ===")

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Frozen Parameters: {frozen_params}")

print(
    f"Trainable Ratio: "
    f"{trainable_params / total_params * 100:.2f}%"
)