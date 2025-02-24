#Gradients for model optimizations

import torch

# we would need torch to calculate gradients
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
# since x was created with requires_grad on, pytorch will provide
# a method to calcuate gradients of functions on x w.r.t. x
print(y)

z = y * y * 2
print(z)
z = z.mean()
print(z)

# call backward gradient function of z w.r.t. x
# z is a single value (mean), therefore backward method can be called without any parameter
z.backward()
# if z is not a scalar value , you have to pass a vector or same size to backward method
# now z would have an attribute grad that would contain gradient values of all values from z w.r.t to
# corresponding values in x
print(x.grad)

# how to prevent torch from recording the gradient history
# x.requires_grad_(False) ----- a function with underscore modifies the attribute with same name
# x.detach()
# wrap inside with torch.no_grad()

## ACCUMULATION OF GRADIENT HISTORY
weights  = torch.ones(4, requires_grad=True)

for epochs in range(2):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()