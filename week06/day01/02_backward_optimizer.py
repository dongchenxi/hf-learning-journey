import torch
from torch import nn


# 1. 固定随机种子
torch.manual_seed(42)


# 2. 定义模型
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


# 3. 冻结 layer1
for param in model.layer1.parameters():
    param.requires_grad = False


# 4. 创建优化器
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.01
)


# 5. 保存训练前的参数
before = {
    name: param
    for name, param in model.named_parameters()
}


# 6. 创建模拟训练数据
x = torch.randn(4, 10)
target = torch.randn(4, 5)


# 7. Forward
output = model(x)


# 8. Loss
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)


# 9. Backward
optimizer.zero_grad()
loss.backward()


print("=== Gradients ===")

for name, param in model.named_parameters():
    print(
        f"{name}: "
        f"grad_is_none={param.grad is None}"
    )


# 10. 更新参数
optimizer.step()


# 11. 比较训练前后的参数
print("\n=== Parameter Changes ===")

for name, param in model.named_parameters():

    changed = not torch.equal(
        before[name],
        param.detach()
    )

    print(f"{name}: changed={changed}")


print(f"\nLoss: {loss.item():.6f}")