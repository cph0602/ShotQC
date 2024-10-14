import itertools, torch
from math import sqrt, floor
from shotqc.helper import params_list_to_matrix, generate_matrix, params_matrix_to_list




def parallel_entry_coef(current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states, params):
    cuts = info["cuts"] # (meas, prep)
    num_qubits = info["num_qubits"]
    num_subcircuits = info["num_subcircuits"]
    
    params_matrix = params_list_to_matrix(params, prep_states)
    coef_matrix = generate_matrix(params_matrix, prep_states)
    # print(coef_matrix)
    
    entry_coef = [[0 for _ in range(len(entry_dict[subcircuit_idx]))] for subcircuit_idx in range(num_subcircuits)]

    prep = [[None for _ in range(subcircuits_info[subcircuit_idx]["num_qubits"])] for subcircuit_idx in range(num_subcircuits)]
    for bittuple in itertools.product("01", repeat = num_qubits):
        bitstring = ''.join(bittuple)
        prob_coef = [{} for _ in range(num_subcircuits)]
        # prob_coef_shape = (num_subcircuits, ) 
        for subcircuit_idx in range(num_subcircuits):
            for prep_config in itertools.product(prep_states, repeat=subcircuits_info[subcircuit_idx]["counter"]["rho"]):
                for meas_config in itertools.product(range(6), repeat=subcircuits_info[subcircuit_idx]["counter"]["O"]):
                    prob_coef[subcircuit_idx][(prep_config, meas_config)] = torch.tensor(0.0, requires_grad=True)
        # collect prob coefficients (D[cost_function, P[some_config]])
        for prep_config in itertools.product(prep_states, repeat = info["num_cuts"]):
            # set prep configurations
            for cut in cuts.keys():
                prep[cuts[cut][1][0]][cuts[cut][1][1]] = prep_config[info["cut_index"][cut]]
            subcircuits_values = []
            bit_countdown = num_qubits - 1
            current_coef = []
            for subcircuit_idx in range(num_subcircuits):
                num_meas = subcircuits_info[subcircuit_idx]["counter"]["O"]
                csc_shape = (6,)*num_meas
                current_subcircuit_coef = torch.zeros(csc_shape, requires_grad=True)
                subcircuit_sum = torch.tensor(0.0, requires_grad=True)
                output_bits = []
                for i in range(subcircuits_info[subcircuit_idx]["num_qubits"]):
                    if subcircuits_info[subcircuit_idx]['output'][i] == None:
                        output_bits.append(bitstring[bit_countdown])
                        bit_countdown -= 1
                    else:
                        output_bits.append(None)
                # print(bitstring, output_bits)
                meas = [None for _ in range(subcircuits_info[subcircuit_idx]["num_qubits"])]
                for meas_config in itertools.product(range(6), repeat=num_meas):
                    zero_coefficient = False
                    coef_product = torch.tensor(1.0, requires_grad=True)
                    for meas_idx, meas_cut in enumerate(meas_config):
                        meas_qubit = list(subcircuits_info[subcircuit_idx]["m_cuts"].keys())[meas_idx]
                        meas_label = subcircuits_info[subcircuit_idx]["m_cuts"][meas_qubit]
                        meas[meas_qubit] = meas_cut // 2
                        output_bits[meas_qubit] = str(meas_cut % 2)
                        tmp_cut_idx = info["cut_index"][meas_label]
                        cut_coef = coef_matrix[tmp_cut_idx][prep_config[tmp_cut_idx]][meas_cut]
                        if cut_coef == 0:
                            zero_coefficient = True
                            break
                        else:
                            coef_product = coef_product * cut_coef
                    if zero_coefficient:
                        continue
                    subcircuit_bitstring = (''.join(output_bits))[::-1]
                    entry_idx = entry_dict[subcircuit_idx][(tuple(prep[subcircuit_idx]), tuple(meas))]
                    prob = current_prob_with_prior[subcircuit_idx][entry_idx][subcircuit_bitstring]
                    subcircuit_sum = subcircuit_sum + prob * coef_product
                    current_subcircuit_coef[*meas_config] = current_subcircuit_coef[*meas_config] + coef_product
                subcircuits_values.append(subcircuit_sum)
                current_coef.append(current_subcircuit_coef)
            current_coef = torch.stack(current_coef)
            subcircuits_values = torch.stack(subcircuits_values)
            total_product = torch.tensor(1.0, requires_grad=True)
            num_zeros = 0
            for subcircuit_idx in range(num_subcircuits):
                if num_zeros == 2:
                    break
                if subcircuits_values[subcircuit_idx] == 0:
                    num_zeros += 1
                    zero_idx = subcircuit_idx
                else:
                    total_product = torch.product * subcircuits_values[subcircuit_idx]
            if num_zeros >= 2:
                continue
            elif num_zeros == 1:
                subcircuit_prep_config = []
                for qubit in range(subcircuits_info[zero_idx]["num_qubits"]):
                    if prep[zero_idx][qubit] != None:
                        subcircuit_prep_config.append(prep[zero_idx][qubit])
                for coef_key in current_coef[zero_idx]:
                    prob_coef[zero_idx][(tuple(subcircuit_prep_config), coef_key)] += current_coef[zero_idx][coef_key] * total_product
            else:
                for subcircuit_idx in range(num_subcircuits):
                    subcircuit_prep_config = []
                    for qubit in range(subcircuits_info[subcircuit_idx]["num_qubits"]):
                        if prep[subcircuit_idx][qubit] != None:
                            subcircuit_prep_config.append(prep[subcircuit_idx][qubit])
                    for coef_key in current_coef[subcircuit_idx]:
                        prob_coef[subcircuit_idx][(tuple(subcircuit_prep_config), coef_key)] += current_coef[subcircuit_idx][coef_key] * total_product / subcircuits_values[subcircuit_idx]

        bit_countdown = num_qubits - 1
        # calculate variance
        for subcircuit_idx in range(num_subcircuits):
            output_bits = []
            for i in range(subcircuits_info[subcircuit_idx]["num_qubits"]):
                if subcircuits_info[subcircuit_idx]['output'][i] == None:
                    output_bits.append(bitstring[bit_countdown])
                    bit_countdown -= 1
                else:
                    output_bits.append(None)

            for subcircuit_prep_config in itertools.product(prep_states, repeat=subcircuits_info[subcircuit_idx]["counter"]["rho"]):
                subcircuit_prep = [None for _ in range(subcircuits_info[subcircuit_idx]["num_qubits"])]
                for prep_idx, prep_cut in enumerate(subcircuit_prep_config):
                    prep_qubit = list(subcircuits_info[subcircuit_idx]["p_cuts"].keys())[prep_idx]
                    subcircuit_prep[prep_qubit] = prep_cut
                for subcircuit_meas_config in itertools.product(range(6), repeat=subcircuits_info[subcircuit_idx]["counter"]["O"]):
                    subcircuit_meas = [None for _ in range(subcircuits_info[subcircuit_idx]["num_qubits"])]
                    for meas_idx, meas_cut in enumerate(subcircuit_meas_config):
                        meas_qubit = list(subcircuits_info[subcircuit_idx]["m_cuts"].keys())[meas_idx]
                        subcircuit_meas[meas_qubit] = meas_cut // 2
                        output_bits[meas_qubit] = str(meas_cut % 2)
                    subcircuit_bitstring = (''.join(output_bits))[::-1]
                    related_bitstrings = []
                    related_meas_configs = []
                    for meas_idx, meas_cut in enumerate(subcircuit_meas_config):
                        if meas_cut % 2 == 0:
                            output_bits_copy = output_bits[:]
                            subcircuit_meas_config_copy = list(subcircuit_meas_config)[:]
                            meas_qubit = list(subcircuits_info[subcircuit_idx]["m_cuts"].keys())[meas_idx]
                            output_bits_copy[meas_qubit] = str(meas_cut % 2 + 1)
                            subcircuit_meas_config_copy[meas_idx] = meas_cut + 1
                            subcircuit_meas_config_copy = tuple(subcircuit_meas_config_copy)
                            related_bitstrings.append((''.join(output_bits_copy))[::-1])
                            related_meas_configs.append(subcircuit_meas_config_copy)
                    entry_idx = entry_dict[subcircuit_idx][(tuple(subcircuit_prep), tuple(subcircuit_meas))]
                    self_coef = prob_coef[subcircuit_idx][(subcircuit_prep_config, subcircuit_meas_config)]
                    self_prob = current_prob_with_prior[subcircuit_idx][entry_idx][subcircuit_bitstring]
                    # calculate self variance
                    entry_coef[subcircuit_idx][entry_idx] += self_prob * (1 - self_prob) * (self_coef**2)
                    # calculate covariance between main string and related strings
                    for relative_idx in range(len(related_bitstrings)):
                        relative_coef = prob_coef[subcircuit_idx][(subcircuit_prep_config, related_meas_configs[relative_idx])]
                        relative_prob = current_prob_with_prior[subcircuit_idx][entry_idx][related_bitstrings[relative_idx]]
                        entry_coef[subcircuit_idx][entry_idx] -= 2 * self_coef * relative_coef * self_prob * relative_prob
        
    return entry_coef


def parallel_cost_function(params, current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states):
    cost = torch.tensor(0.0, requires_grad=True)
    coefficients = parallel_entry_coef(current_prob_with_prior, entry_dict, info, subcircuits_info, prep_states, params)
    for subcircuit_idx in range(info["num_subcircuits"]):
        for entry_idx in range(len(entry_dict[subcircuit_idx])):
            cost = cost + torch.sqrt(coefficients[subcircuit_idx][entry_idx])
    return cost


