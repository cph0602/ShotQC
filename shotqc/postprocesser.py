import itertools, pickle
from shotqc.helper import generate_prob_from_counts_with_prior, generate_matrix


def read_probs(data_folder):
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    subcircuits = meta_info["subcircuits"]
    probs = []
    for subcircuit_idx in range(len(subcircuits)):
        subcircuit_entries = []
        for entry_idx in range(len(list(entry_dict[subcircuit_idx].keys()))):
            counts = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, entry_idx), "rb"))
            prob_read = generate_prob_from_counts_with_prior(counts)
            subcircuit_entries.append(prob_read)
        probs.append(subcircuit_entries)
    return probs


def postprocess(data_folder: str, info, subcircuits_info, prep_states, params):
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    subcircuits = meta_info["subcircuits"]

    cuts = info["cuts"] # (meas, prep)
    num_qubits = info["num_qubits"]
    # initialize probabilities
    output_prob = {}
    prep = [[None for _ in range(subcircuit.num_qubits)] for subcircuit in subcircuits]
    for bittuple in itertools.product("01", repeat = num_qubits):
        bitstring = ''.join(bittuple)
        output_prob[bitstring] = 0
    coef_matrix = generate_matrix(params, prep_states)
    # print(coef_matrix)
    probs = read_probs(data_folder)
    for bittuple in itertools.product("01", repeat = num_qubits):
        bitstring = ''.join(bittuple)
        for prep_config in itertools.product(prep_states, repeat = info["num_cuts"]):
            # set prep configurations
            for cut in cuts.keys():
                prep[cuts[cut][1][0]][cuts[cut][1][1]] = prep_config[info["cut_index"][cut]]
            # postprocess
            subcircuits_values = []
            bit_countdown = num_qubits - 1
            for subcircuit_idx in range(len(subcircuits)):
                subcircuit_sum = 0
                num_meas = subcircuits_info[subcircuit_idx]["counter"]["O"]
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
                    coef_product = 1
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
                            coef_product *= cut_coef
                    if zero_coefficient:
                        continue
                    subcircuit_bitstring = (''.join(output_bits))[::-1]
                    entry_idx = entry_dict[subcircuit_idx][(tuple(prep[subcircuit_idx]), tuple(meas))]
                    prob = probs[subcircuit_idx][entry_idx][subcircuit_bitstring]
                    subcircuit_sum += prob * coef_product
                subcircuits_values.append(subcircuit_sum)
            total_product = 1
            for subcircuit_idx in range(len(subcircuits)):
                total_product *= subcircuits_values[subcircuit_idx]
            output_prob[bitstring] += total_product.item()
    return output_prob
