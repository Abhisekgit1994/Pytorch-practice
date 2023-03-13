import torch

batch = 10
features = 25
x = torch.rand((batch, features))

print(x[0].shape) # x[0,:]

print(x[:, 0].shape)

print(x[2, 0:10].shape)

print(x[0, 0])
print(x[0][0])
idx = [2, 4, 8]
print(x[idx])

x= torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

# x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[x.remainder(2)==0])

# useful operatons
print(torch.where(x>5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())
print(x.numel())