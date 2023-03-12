import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_ten = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

print(my_ten)
print(my_ten.dtype)
print(my_ten.device)
print(my_ten.shape)

# Other initialization methods:
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(5, 5)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# tensor typecasting

my_ten = torch.arange(4)
print(my_ten.bool())
print(my_ten.long())
print(my_ten.half())
print(my_ten.float())

# Array to tensor amd tensor to array
arr = np.zeros((5, 5))
my_ten = torch.from_numpy(arr)
arr = my_ten.numpy()
print(arr)


