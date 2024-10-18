import torch, random, itertools
from time import perf_counter
import torch

a = torch.tensor([[1.,2.,3.], [4.,5.,6.]], requires_grad=True)
print(a[:,1:3])
a[:,1:3] = a[:, 1:3] + torch.ones((2,2))
print(a)