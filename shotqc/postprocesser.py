import itertools, pickle
from scipy.optimize import minimize

def generate_matrix_from_params(params, prep_states):
    """
    return List[cut_idx][init_state][meas_basis]
    """
    M = []
    for param in params:
        M_cut = []
        if prep_states == [0, 2, 4, 5]:
            assert len(param) == 8
            # param = [a1, a3, a5, a6, b1, b3, b5, b6]
            M_cut = [
                [2-param[0], -2-param[0], -param[4], -param[4], param[0]+param[4], param[0]+param[4]],
                [0 for _ in range(6)],
                [-param[1], -param[1], 2-param[5], -2-param[5], param[1]+param[5], param[1]+param[5]],
                [0 for _ in range(6)],
                [-1-param[2], 1-param[2], -1-param[6], 1-param[6], 2+param[2]+param[6], param[2]+param[6]],
                [-1-param[3], 1-param[3], -1-param[7], 1-param[7], param[3]+param[7], param[3]+param[7]]
            ]
        elif prep_states == range(6):
            assert len(param) == 24
            a = param[:6]
            b = param[6:12]
            c = param[12:18]
            d = param[18:]
            M_cut = [
                [1-a[0]-c[0], -1-a[0]-c[1], -b[0]-c[2], -b[0]-c[3], a[0]+b[0]-c[4], a[0]+b[0]-c[5]],
                [-1-a[1]-c[0], 1-a[1]-c[1], -b[1]-c[2], -b[1]-c[3], a[1]+b[1]-c[4], a[1]+b[1]-c[5]],
                [-a[2]-d[0], -a[2]-d[1], 1-b[2]-d[2], -1-b[2]-d[3], a[2]+b[2]-d[4], a[2]+b[2]-d[5]],
                [-a[3]-d[0], -a[3]-d[1], -1-b[3]-d[2], 1-b[3]-d[3], a[3]+b[3]-d[4], a[3]+b[3]-d[5]],
                [-a[4]+c[0]+d[0], -a[4]+c[1]+d[1], -b[4]+c[2]+d[2], -b[4]+c[3]+d[3], 2+a[4]+b[4]+c[4]+d[4], a[4]+b[4]+c[5]+d[5]],
                [-a[5]+c[0]+d[0], -a[5]+c[1]+d[1], -b[5]+c[2]+d[2], -b[5]+c[3]+d[3], a[5]+b[5]+c[4]+d[4], 2+a[5]+b[5]+c[5]+d[5]]
            ]
        else:
            raise Exception("current state set not supported")
        M_cut = [[M_cut[i][j]/2 for j in range(6)] for i in range(6)]
        M.append(M_cut)
    return M


def generate_prob_from_counts(counts):
    total_num = counts["total_shots"]
    prob = {}
    for key in counts["counts"].keys():
        prob[key] = counts["counts"][key] / total_num
    return prob


def postprocess(data_folder: str, info, subcircuits_info, prep_states, params):
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    subcircuits = meta_info["subcircuits"]

    cuts = info["cuts"] # (meas, prep)
    num_qubits = info["num_qubits"]
    # initialize probabilities
    output_prob = {}
    for bittuple in itertools.product("01", repeat = num_qubits):
        bitstring = ''.join(bittuple)
        output_prob[bitstring] = 0
    coef_matrix = generate_matrix_from_params(params, prep_states)

    for prep_config in itertools.product(prep_states, repeat = info["num_cuts"]):
        # set prep configurations
        prep = [[None for _ in range(subcircuit.num_qubits)] for subcircuit in subcircuits]
        for cut in cuts.keys():
            prep[cuts[cut][1][0]][cuts[cut][1][1]] = prep_config[info["cut_index"][cut]]
        
        # set and collect all measurement configurations 
        meas = [[None for _ in range(subcircuit.num_qubits)] for subcircuit in subcircuits]
        prob_dict = [{} for _ in range(len(subcircuits))]
        for meas_config in itertools.product(range(3), repeat = info["num_cuts"]):
            for cut in cuts.keys():
                meas[cuts[cut][0][0]][cuts[cut][0][1]] = meas_config[info["cut_index"][cut]]
            for subcircuit_idx in range(len(subcircuits)):
                entry_idx = entry_dict[subcircuit_idx][(tuple(prep[subcircuit_idx]), tuple(meas[subcircuit_idx]))]
                entry_count = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, entry_idx), "rb"))
                prob_dict[subcircuit_idx][tuple(meas[subcircuit_idx])] = generate_prob_from_counts(entry_count)

        # postprocess
        
        for bittuple in itertools.product("01", repeat = num_qubits):
            bitstring = ''.join(bittuple)
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
                    prob = prob_dict[subcircuit_idx][tuple(meas)][subcircuit_bitstring]
                    subcircuit_sum += prob * coef_product
                subcircuits_values.append(subcircuit_sum)
            total_product = 1
            for subcircuit_idx in range(len(subcircuits)):
                total_product *= subcircuits_values[subcircuit_idx]
            output_prob[bitstring] += total_product
    
    print(output_prob)
