import torch
import torch.nn.functional as F

a = torch.Tensor([[1,2, 3],[4,5, 6]])

print(a)

a = a.transpose(-1, -2)

a = a.unsqueeze(1)

b = F.avg_pool1d(a, kernel_size=a.size(-1)).squeeze(1).squeeze(-1)

print(b)
