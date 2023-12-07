import torch

X = torch.tensor([[1.],[2.],[3.]])
x2 = torch.tensor([2.], requires_grad=True)
x4 = torch.tensor([4.], requires_grad=True)
y = torch.tensor([[5.],[10.],[15.]])


print(x4+(1./x2)-X)
loss = 0.5/3 * ((y-(5. + 10.*(x4 + x2*X)))**2).sum()

loss.backward()


print(x4.grad)
print(x2.grad)

