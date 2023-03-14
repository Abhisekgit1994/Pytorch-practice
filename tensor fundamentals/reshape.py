# tensor reshaping
import torch

x = torch.rand(9)
x_r = x.view(3,3)  # view act on contiguous tensors
x_r = x.reshape(3, 3)
print(x_r)

y = x_r.t()
print(y)
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(x1)
print(x2)
print(torch.cat((x1, x2), dim=1))

print(x1.view(-1))

batch = 16
x = torch.rand((batch, 2, 5))
z = x.view(batch,-1)    # keep the batch dimension and flatten the rest
print(z.shape)
z = x.permute(0, 2, 1)   # alter the dimensions
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0))
print(x.unsqueeze(1))

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x)
print(x.shape)

print(x.squeeze(1))

x = torch.rand((64,1,28,28))
print(x.squeeze(1).shape)

x = torch.rand((4, 64,256))
print(x[-1].shape)
print(torch.cat((x[-1], x[-2]), dim=1).shape)


