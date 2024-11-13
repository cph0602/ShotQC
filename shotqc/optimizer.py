from shotqc.parallel_overhead_v2 import parallel_cost_function, parallel_variance, total_entry_coef, batch_loss, batch_variance
import torch.optim as optim
from shotqc.helper import bitstring_batch_generator
from math import ceil
import torch
from time import perf_counter
from torch.optim.lr_scheduler import MultiStepLR


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

def parallel_minimize_var(init_params, args, shot_count, lr=0.01, num_iterations=100, device=None, batch_size=1024):
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

def batch_optimize_params(init_params, args, lr=0.1, num_iterations=100, device=None, batch_size=1024):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    for i in range(num_iterations):
        iter_start = perf_counter()
        optimizer.zero_grad()  # Clear previous gradients
        num_batch = ceil(2**args.num_qubits/batch_size)
        batch_cnt = 0
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
            batch_cnt += 1
            print(f"Batch {batch_cnt}/{num_batch} done. Mem used: {torch.cuda.memory_allocated(device)}")
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
            print(f"Step {i+1}/{num_iterations}: Cost = {current_loss}; Iteration time = {(perf_counter()-iter_start)/60} mins")
        else:
            print(f"Step {i+1}/{num_iterations}: Iteration time = {(perf_counter()-iter_start)/60} mins")
    return current_loss, params

def batch_minimize_var(init_params, args, shot_count, lr=0.1, num_iterations=100, device=None, batch_size=1024):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[70], gamma=0.1)
    for i in range(num_iterations):
        iter_start = perf_counter()
        optimizer.zero_grad()  # Clear previous gradients
        num_batch = ceil(2**args.num_qubits/batch_size)
        batch_cnt = 0
        current_loss = 0
        total_var = 0
        for batch in bitstring_batch_generator(args.num_qubits, batch_size):
            var = batch_variance(
                params=params,
                args=args,
                shot_count=shot_count,
                batch=batch,
                device=device
            )  # Compute the cost function
            var.backward()  # Compute gradients
            total_var += var.detach().cpu().item()
            batch_cnt += 1
            print(f"Batch {batch_cnt}/{num_batch} done. Mem used: {torch.cuda.memory_allocated(device)}")
            del var
        optimizer.step()  # Update x using the optimizer
        scheduler.step()
        print(f"Step {i+1}/{num_iterations}: Variance = {total_var}; Iteration time = {(perf_counter()-iter_start)/60} mins")
    return current_loss, params