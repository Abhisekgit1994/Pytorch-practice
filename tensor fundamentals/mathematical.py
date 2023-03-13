# mathematical and comparison operations in torch

import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([6, 7, 8])

z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x+y
print(z)

# Division
z = torch.true_divide(x,y)
print(z)

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x
print(t)

z = x.pow(2)
print(z)

z = x > 1
print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
print(x3)
x3 = x1.mm(x2)
print(x3)

# Matrix power
matrix = torch.rand(5, 5)
print(matrix.matrix_power(2))

# element wise multiplication
z = x * y
print(z)

z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n =10
m = 20
p = 30
t1 = torch.rand((batch, n , m))
t2 = torch.rand((batch, m , p))
out = torch.bmm(t1, t2)
print(out.shape)  # batch , m ,p

# broadcasting
# t1 = torch.rand((5, 5))
# t2 = torch.rand((1, 5))
# z = t1 - t2
# print(z)

print(x1)
sum_x = torch.sum(x1, dim=0)
print(sum_x, sum_x.shape)
values, idx = torch.max(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
print(z)

print(torch.sum(t1[0], dim=0))
sum_x = torch.sum(t1, dim=0)
print(sum_x.shape)

print(torch.eq(x, y))

print(torch.sort(y, dim=0, descending=False))
z = torch.clamp(x, min=2, max=10)
print(z)

