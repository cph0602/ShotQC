import pickle, random
from scipy.optimize import minimize, approx_fprime
import numpy as np
from time import perf_counter
from math import ceil
from shotqc.helper import params_list_to_matrix, read_probs_with_prior
from shotqc.overhead import cost_function



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


def optimize_params(data_folder, info, subcircuits_info, prep_states, prior, method):
    if prep_states == [0,2,4,5]:
        num_params = info["num_cuts"] * 8
    elif prep_states == range(6):
        num_params = info["num_cuts"] * 24
    else:
        raise Exception("initial state set not supported")
    initial_guess = np.zeros(num_params)
    for i in range(num_params):
        initial_guess[i] = random.uniform(-2, 2)
    boundaries = [(-2, 2)] * num_params
    current_prob_with_prior = read_probs_with_prior(data_folder, prior)
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    one_eval_time = perf_counter()
    print("cost w/o optimization: ", cost_function(initial_guess, current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states))
    one_eval_time = perf_counter() - one_eval_time
    args = (current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states)
    if method == "L-BFGS-B":
        options = {'maxiter': 100}
        minimize_time = perf_counter()
        result = minimize(cost_function, initial_guess, bounds=boundaries, args=args, options=options, method="L-BFGS-B")
        minimize_time = perf_counter() - minimize_time
        print(f"Time spent in minimization: {minimize_time} seconds")
        return(result.fun, params_list_to_matrix(result.x, prep_states))
    elif method == "Adam":
        maxiter = ceil(60 / (one_eval_time * info["num_total_entries"]))
        print("Maxiter: ", maxiter)
        minimize_time = perf_counter()
        new_params, minvar = adam_optimizer(cost_function, initial_guess, *args, max_iter=maxiter)
        minimize_time = perf_counter() - minimize_time
        print(f"Time spent in minimization: {minimize_time} seconds")
        return (minvar, params_list_to_matrix(new_params, prep_states))
    else:
        raise Exception("minimization method not supported")