from shotqc.parallel_overhead_v2 import parallel_cost_function, parallel_variance
import torch.optim as optim


def parallel_optimize_params_sgd(init_params, args, lr=0.01, momentum=0.9, num_iterations=100, device=None, batch_size=1024):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        loss = parallel_cost_function(
            params=params,
            args=args,
            device=device,
            batch_size=batch_size,
            verbose=True
        )  # Compute the cost function
        loss.backward()  # Compute gradients
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            print(f"Step {i+1}/{num_iterations}: Cost = {loss.item()}")
    return loss.item(), params

def parallel_minimize_var(init_params, args, shot_count, lr=0.1, num_iterations=100, device=None, batch_size=1024):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        loss = parallel_variance(
            params=params,
            args=args,
            shot_count=shot_count,
            device=device,
            batch_size=batch_size,
            verbose=False
        )  # Compute the cost function
        loss.backward()  # Compute gradients
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            print(f"Step {i+1}/{num_iterations}: Cost = {loss.item()}")
    return loss.item(), params
