import itertools, pickle
from scipy.optimize import minimize

def optimize_params(data_folder: str, info, subcircuits_info, prep_states):
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    subcircuits = meta_info["subcircuits"]
    flag = True
    cuts = info["cuts"] # (meas, prep)
    for prep_config in itertools.product(prep_states, repeat = info["num_cuts"]):
        prep = [[None for _ in range(subcircuit.num_qubits)] for subcircuit in subcircuits]
        for index, cut in enumerate(list(cuts.keys())):
            prep[cuts[cut][1][0]][cuts[cut][1][1]] = prep_config[index]
        meas = [[None for _ in range(subcircuit.num_qubits)] for subcircuit in subcircuits]
        count_dict = [{} for _ in range(len(subcircuits))]
        for meas_config in itertools.product(range(3), repeat = info["num_cuts"]):
            for index, cut in enumerate(list(cuts.keys())):
                meas[cuts[cut][0][0]][cuts[cut][0][1]] = meas_config[index]
            for subcircuit_idx in range(len(subcircuits)):
                entry_idx = entry_dict[subcircuit_idx][(tuple(prep[subcircuit_idx]), tuple(meas[subcircuit_idx]))]
                entry_count = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, entry_idx), "rb"))
                count_dict[subcircuit_idx][tuple(meas[subcircuit_idx])] = entry_count
        