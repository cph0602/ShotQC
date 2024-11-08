import itertools, torch, pickle, time, os, subprocess
from math import sqrt, floor, ceil
from shotqc.helper import (params_list_to_matrix, generate_matrix, params_matrix_to_list, 
                           tensor_product, bitstring_batch_generator)
from time import perf_counter

def calc_subcircuit_value(coef_matrix, args, batch, prep_config, subcircuit_idx, device):
    # calculate coef tensor product with gradient
    #  *. Misc Variables
    # calc_start = perf_counter()
    num_meas = args.num_meas[subcircuit_idx]
    batch_size = batch.shape[0]
    #  0. Find base local entry idx: entries from base -> base + 3^num_meas -1 are used
    base_local_entry_idx = 0
    for prep_cut in args.prep_cuts[subcircuit_idx]:
        base_local_entry_idx *= args.len_prep_states
        base_local_entry_idx += prep_config[prep_cut]
    base_local_entry_idx *= 3**num_meas
    #  1. fetch coef matrix lines that correspond to measurement cut indexes
    coef_matrix_rows = []
    for meas_cut in args.meas_cuts[subcircuit_idx]:
        coef_matrix_rows.append(coef_matrix[meas_cut][args.prep_states[prep_config[meas_cut]]][:])
    #  2. perform tensor product
    # moved back
    #  3. find correct section of entry_probs that correspond to this prep_config
    subcircuit_bitstrings = torch.flip(batch, dims=[1])[:, args.acc_eff_qubits[subcircuit_idx]:args.acc_eff_qubits[subcircuit_idx+1]]
    indices = tuple(subcircuit_bitstrings[:, i] for i in range(subcircuit_bitstrings.shape[1]))
    padded_permute_order = torch.cat((torch.tensor([0],device=device), args.permute_orders[subcircuit_idx] + torch.ones_like(args.permute_orders[subcircuit_idx], device=device)))
    permuted_probs = torch.permute(args.entry_probs[subcircuit_idx][base_local_entry_idx:base_local_entry_idx + 3**num_meas], tuple(padded_permute_order)).contiguous()
    final_permute_order = (1,0) + tuple([i+2 for i in range(num_meas)])
    used_probs = permuted_probs[(slice(None),) + indices].permute(final_permute_order).contiguous()
    #  4. perform element-wise multiplication
    #    (a) lining them up
    coef_permute_tuple = tuple([i*2 for i in range(num_meas)]) + tuple([i*2+1 for i in range(num_meas)])
    coef_tensor_product = tensor_product(coef_matrix_rows, device)
    lined_coef_products = coef_tensor_product.view((3,2)*num_meas).permute(coef_permute_tuple).contiguous().view((3**num_meas,)+(2,)*num_meas)
    #    (b) multiply
    #  4.5. use sum() for subcircuit value
    sum_dimensions = tuple([i+1 for i in range(num_meas+1)])
    subcircuit_value = torch.sum(used_probs * lined_coef_products, dim=(sum_dimensions))
    #  5. return result, and remember to tell main function which section it is
    # print("time used here: ", perf_counter()-calc_start)
    return subcircuit_value, lined_coef_products # shape = (3**num_meas, ) + (2,)*num_meas

def total_entry_coef(params, args, device=None, batch_size=1024, verbose=True):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## main
    params = params.to(device)
    params_matrix = params_list_to_matrix(params, args.prep_states)
    coef_matrix = generate_matrix(params_matrix, args.prep_states)
    # print(coef_matrix)
    entry_coef = [torch.zeros(args.num_entries[subcircuit_idx], device=device) for subcircuit_idx in range(args.num_subcircuits)]
    # total_values = torch.tensor([], device=device) # for reconstructing
    batch_cnt = 0
    num_batch = ceil(2**args.num_qubits/batch_size)
    for batch in bitstring_batch_generator(args.num_qubits, batch_size):
        start_time = perf_counter()
        # Calculate prob_coef
        this_batch_size = batch.shape[0]
        # this_total_value = torch.zeros(this_batch_size, device=device)
        prob_coef = [
            torch.zeros((this_batch_size,args.num_entries[subcircuit_idx])+(2,)*args.num_meas[subcircuit_idx],
            requires_grad=True, device=device)
            for subcircuit_idx in range(args.num_subcircuits)
        ] # size = batch_size * num_entries * 2**num_meas
        for prep_config in itertools.product(range(args.len_prep_states), repeat=args.num_cuts):
            value = torch.ones(this_batch_size, device=device)
            subcircuit_values = []
            config_prob_coef = []
            for subcircuit_idx in range(args.num_subcircuits):
                # going through a batch of entries
                subcircuit_value, lined_prob_coef = calc_subcircuit_value(
                    coef_matrix, args, batch, prep_config, subcircuit_idx, device
                )
                subcircuit_values.append(subcircuit_value)
                config_prob_coef.append(lined_prob_coef) # this stores each cut outcome for a number of entries.
                value = value * subcircuit_value
            # this_total_value = this_total_value + value
            subcircuit_values = torch.stack(subcircuit_values)
            product_except_self = [torch.ones(this_batch_size, device=device) for subcircuit_idx in range(args.num_subcircuits)]
            # print(config_prob_coef[1])
            for subcircuit_idx in range(args.num_subcircuits):
                mask = torch.ones(args.num_subcircuits, dtype=bool)
                mask[subcircuit_idx] = False
                product_except_self[subcircuit_idx] = torch.prod(subcircuit_values[mask], dim=0) #(batch_size)
                # if subcircuit_idx == 1:
                #     print(product_except_self[subcircuit_idx])
                # print(config_prob_coef[subcircuit_idx].shape)
                base_local_entry_idx = 0
                for prep_cut in args.prep_cuts[subcircuit_idx]:
                    base_local_entry_idx *= args.len_prep_states
                    base_local_entry_idx += prep_config[prep_cut]
                base_local_entry_idx *= 3**args.num_meas[subcircuit_idx]
                container = torch.zeros_like(prob_coef[subcircuit_idx])
                container[:, base_local_entry_idx:base_local_entry_idx+3**args.num_meas[subcircuit_idx]] = torch.outer(product_except_self[subcircuit_idx], config_prob_coef[subcircuit_idx].flatten()).reshape((this_batch_size,)+config_prob_coef[subcircuit_idx].shape)
                # print(container[:,base_local_entry_idx:base_local_entry_idx+3**args.num_meas[subcircuit_idx]])
                prob_coef[subcircuit_idx] = prob_coef[subcircuit_idx] + container
                del container
            del config_prob_coef
            del product_except_self
            del subcircuit_values
            del lined_prob_coef
            del value
        # total_values = torch.cat((total_values, this_total_value))
        # print(f"--------> Calculate took: {perf_counter()-start_time} seconds")
        # Calculate variance
        for subcircuit_idx in range(args.num_subcircuits):
            num_meas = args.num_meas[subcircuit_idx]
            for entry_idx in range(args.num_entries[subcircuit_idx]):
                # 1. Self variance
                subcircuit_bitstrings = torch.flip(batch, dims=[1])[:, args.acc_eff_qubits[subcircuit_idx]:args.acc_eff_qubits[subcircuit_idx+1]]
                indices = tuple(subcircuit_bitstrings[:, i] for i in range(subcircuit_bitstrings.shape[1]))
                padded_permute_order = args.permute_orders[subcircuit_idx]
                permuted_probs = torch.permute(args.entry_probs[subcircuit_idx][entry_idx], tuple(padded_permute_order)).contiguous()
                self_probs = permuted_probs[indices]
                one_minus_probs = torch.ones_like(self_probs, requires_grad=True) - self_probs
                self_coefs = prob_coef[subcircuit_idx][:, entry_idx]
                # print(self_coefs)
                variance = torch.sum(self_probs*one_minus_probs*self_coefs*self_coefs, dim=tuple(range(1,num_meas+1)))
                # print(variance)
                # 2. Covariance
                #   (a) Find all relative bitstrings
                if num_meas != 0:
                    outer_probs = self_probs.view(this_batch_size, -1, 1) * self_probs.view(this_batch_size, 1, -1)
                    outer_coefs = self_coefs.view(this_batch_size, -1, 1) * self_coefs.view(this_batch_size, 1, -1)
                    covariance = torch.sum(self_probs*self_probs*self_coefs*self_coefs, dim=tuple(range(1,num_meas+1))) - torch.sum(outer_coefs*outer_probs, dim=(1,2))
                    entry_coef[subcircuit_idx][entry_idx] = entry_coef[subcircuit_idx][entry_idx] + torch.sum(variance + covariance)
                    del outer_coefs, outer_probs, covariance
                else:
                    entry_coef[subcircuit_idx][entry_idx] = entry_coef[subcircuit_idx][entry_idx] + torch.sum(variance)
                del self_coefs, self_probs
                del one_minus_probs
                del variance
                del subcircuit_bitstrings
                del permuted_probs
                del indices, padded_permute_order
        if verbose:
            print(f"-----> Batch {batch_cnt+1}/{num_batch} Completed, used {perf_counter()-start_time} seconds")
        batch_cnt += 1
        del prob_coef
    return entry_coef

def parallel_cost_function(params, args, device=None, batch_size=1024, verbose=False):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entry_coef = total_entry_coef(
        params=params,
        args=args,
        device=device,
        batch_size=batch_size,
        verbose=verbose
    )
    cost = torch.tensor(0., requires_grad=True, device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        cost = cost + torch.sum(torch.sqrt(torch.clamp(entry_coef[subcircuit_idx], min=1e-8)))
    return cost

def parallel_distribute(params, args, total_samples, device=None, batch_size=1024, verbose=True):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        entry_coef = total_entry_coef(
            params=params, 
            args=args,
            device=device,
            batch_size=batch_size,
            verbose=verbose
        )
    total_cost = torch.tensor(0., device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        total_cost = total_cost + torch.sum(torch.sqrt(entry_coef[subcircuit_idx]))
    distributions = []
    sample = torch.tensor(total_samples, device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        subcircuit_distribution = torch.floor(sample * torch.sqrt(entry_coef[subcircuit_idx]) / total_cost)
        distributions.append(subcircuit_distribution.tolist())
    # print(distributions)
    return distributions

def parallel_variance(params, args, shot_count, device=None, batch_size=1024, verbose=True):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entry_coef = total_entry_coef(
        params=params,
        args=args,
        device=device,
        batch_size=batch_size,
        verbose=verbose
    )
    var = torch.tensor(0., requires_grad=True, device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        temp = torch.tensor(shot_count[subcircuit_idx], device=device)
        var = var + torch.sum(entry_coef[subcircuit_idx] / temp)
    return var

def parallel_reconstruct(params, args, device=None, batch_size=1024, ext_save=True, verbose=True):
    # print(batch_size)
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## main
    params = params.to(device)
    params_matrix = params_list_to_matrix(params, args.prep_states)
    coef_matrix = generate_matrix(params_matrix, args.prep_states)
    # total_values = torch.tensor([], device=device) # for reconstructing
    batch_cnt = 0
    num_batch = ceil(2**args.num_qubits/batch_size)
    total_values = torch.tensor([], device=device)
    for batch in bitstring_batch_generator(args.num_qubits, batch_size):
        start_time = perf_counter()
        # Calculate prob_coef
        this_batch_size = batch.shape[0]
        this_total_value = torch.zeros(this_batch_size, device=device)
        for prep_config in itertools.product(range(args.len_prep_states), repeat=args.num_cuts):
            value = torch.ones(this_batch_size, device=device)
            for subcircuit_idx in range(args.num_subcircuits):
                # going through a batch of entries
                subcircuit_value, lined_prob_coef = calc_subcircuit_value(
                    coef_matrix, args, batch, prep_config, subcircuit_idx, device
                )
                value = value * subcircuit_value
            this_total_value = this_total_value + value
            del lined_prob_coef
            del subcircuit_value
            del value
        total_values = torch.cat((total_values, this_total_value))
        if verbose:
            print(f"-----> Batch {batch_cnt+1}/{num_batch} Completed, used {perf_counter()-start_time} seconds")
        batch_cnt += 1
    result = {}
    idx = 0
    torch.save(total_values, f'{args.data_folder}/output_tensor.pt')
    for bittuple in itertools.product("01", repeat=args.num_qubits):
        bitstring = ''.join(bittuple)
        result[bitstring] = total_values[idx].item()
        idx += 1
    return result