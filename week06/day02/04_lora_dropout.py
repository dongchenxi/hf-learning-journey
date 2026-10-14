import torch
from torch import nn

torch.manual_seed(42)

dropout = nn.Dropout(p=0.5)

x = torch.ones(20)

print("Original input:")
print(x)

dropout.train()

print("\nTraining mode:")
print(dropout(x))
print(dropout(x))

dropout.eval()

print("\nEvaluation mode:")
print(dropout(x))
print(dropout(x))