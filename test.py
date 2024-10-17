import torch, random, itertools
from time import perf_counter
import torch

def f(x,y):
    # Define a simple forward pass with intermediate results
    

    # Some computation that generates an intermediate tensor
    intermediate_result = x + y

    # Save the intermediate result to disk
    torch.save(intermediate_result, 'intermediate_result.pt')

    # Load it back when needed
    loaded_intermediate = torch.load('intermediate_result.pt').to('cuda')

    # Continue with further computation
    output = torch.sum(loaded_intermediate * 2)
    return output
# print(output.requires_grad)
x = torch.randn(10, 10, requires_grad=True).to('cuda').detach().requires_grad_(True)
y = torch.randn(10, 10).to('cuda')
optimizer = torch.optim.SGD([x])
for i in range(100):
    optimizer.zero_grad()
    output = f(x,y)
    output.backward()
    optimizer.step()
    print(output.item())
