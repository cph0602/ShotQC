import pickle, random, torch, math
from scipy.optimize import minimize, approx_fprime
import numpy as np
from time import perf_counter
from math import ceil
from shotqc.helper import params_list_to_matrix
from shotqc.overhead import cost_function
from shotqc.parallel_overhead_v2 import parallel_cost_function, parallel_variance, parallel_cost_function_grad
import torch.optim as optim



def adam_optimizer(fun, x0, *args, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    for _ in range(max_iter):
        t += 1
        # Compute the gradient (numerical approximation)
        grad = approx_fprime(x, fun, epsilon, *args)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad**2)

        # Correct bias in first and second moment
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return x, fun(x, *args)

def sa_optimizer(fun, x0, *args, initial_temp=10, final_temp=1e-3, alpha=0.99, max_iter=1000):
    bounds = (-2, 2)
    dim = len(x0)
    current_solution = torch.from_numpy(x0)
    current_value = fun(x0, *args)
    for i in range(max_iter):
        temp = initial_temp * (alpha ** i)
        if temp < final_temp:
            break
        
        # Propose a new solution by perturbing the current one
        new_solution = current_solution + 0.1 * torch.randn(dim)
        
        # Apply bounds
        new_solution = torch.clamp(new_solution, bounds[0], bounds[1])
        
        new_value = fun(new_solution, *args)
        
        # Compute acceptance probability
        delta_value = new_value - current_value
        if delta_value < 0 or random.random() < math.exp(-delta_value / temp):
            current_solution = new_solution
            current_value = new_value

        print(f"Iteration {i}: Temp {temp:.4f}, Function Value {current_value}")
    return current_solution, fun(current_solution, *args)


def parallel_cost(x_tensor, current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states):
    # x_numpy = x_tensor.detach().numpy()
    result = cost_function(x_tensor, current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states)
    return torch.tensor(result, dtype=torch.float32)


def parallel_optimize_params_sgd(init_params, args, lr=0.01, momentum=0.9, num_iterations=100, batch_size=64):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([params], lr=lr, momentum=momentum)
    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        loss = parallel_cost_function_grad(params, args, batch_size)  # Compute the cost function
        loss.backward()  # Compute gradients
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            print(f"Step {i+1}/{num_iterations}: Cost = {loss.item()}")
    return loss.item(), params

def parallel_minimize_var(init_params, args, shot_count, lr=0.01, momentum=0.9, num_iterations=100):
    params = init_params.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([params], lr=lr, momentum=momentum)
    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        loss = parallel_variance(params, args, shot_count)  # Compute the cost function
        loss.backward()  # Compute gradients
        optimizer.step()  # Update x using the optimizer
        if (i+1)%10 == 0:
            print(f"Step {i+1}/{num_iterations}: Cost = {loss.item()}")
    return loss.item(), params


