from shotqc.parallel_overhead_v2 import parallel_cost_function, parallel_variance, total_entry_coef, batch_loss
import torch.optim as optim
from shotqc.helper import bitstring_batch_generator
from math import ceil
import torch


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
            verbose=False
        )  # Compute the cost function
        loss.backward()  # Compute gradients
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            print(f"Step {i+1}/{num_iterations}: Cost = {loss.item()}")
        current_loss = loss.detach().cpu().item()
        del loss
    return current_loss, params

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
            print(f"Step {i+1}/{num_iterations}: Variance = {loss.item()}")
        current_loss = loss.detach().cpu().item()
        del loss
    return current_loss, params

def batch_optimize_params(init_params, args, lr=0.01, num_iterations=100, device=None, batch_size=1024):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        num_batch = ceil(2**args.num_qubits/batch_size)
        current_loss = 0
        for batch in bitstring_batch_generator(args.num_qubits, batch_size):
            loss = batch_loss(
                params=params,
                args=args,
                batch=batch,
                device=device,
                batch_size=batch_size,
                verbose=False
            )  # Compute the cost function
            loss.backward()  # Compute gradients
            print(f"batch done. Mem used: {torch.cuda.memory_allocated()}")
            del loss
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            with torch.no_grad():
                current_loss = parallel_cost_function(
                    params=params,
                    args=args,
                    device=device,
                    batch_size=batch_size,
                    verbose=False
                )
            print(f"Step {i+1}/{num_iterations}: Cost = {current_loss}")
    return current_loss, params