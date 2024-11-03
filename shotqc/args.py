import itertools, torch, pickle, time, os, subprocess
from math import sqrt, floor, ceil
from shotqc.helper import (params_list_to_matrix, generate_matrix, params_matrix_to_list, 
                           tensor_product, find_slices, generate_all_bitstrings, 
                           generate_relative_bitstrings, bitstring_batch_generator)
from time import perf_counter

class Args:
    def __init__(self, shotqc, device=None, prior=0):
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.data_folder = shotqc.tmp_data_folder
        meta_info = pickle.load(open("%s/meta_info.pckl" % (self.data_folder), "rb"))
        self.entry_dict = meta_info["entry_dict"]
        self.subcircuits = shotqc.subcircuits
        self.num_subcircuits = len(self.subcircuits)
        self.num_qubits = shotqc.info["num_qubits"]
        self.num_cuts = shotqc.info["num_cuts"]
        self.info = shotqc.info
        self.subcircuits_info = shotqc.subcircuits_info
        self.verbose = shotqc.verbose
        self.gen_misc_infos()
        # print(self.entry_probs[0][0])
        # print(self.num_total_entries)
        # self.gen_output_string()
        # print(self.output_string)
        self.prep_states = shotqc.prep_states
        self.len_prep_states = len(self.prep_states)
        self.num_prep = [shotqc.subcircuits_info[subcircuit_idx]["counter"]['rho'] for subcircuit_idx in range(self.num_subcircuits)]
        self.num_meas = [shotqc.subcircuits_info[subcircuit_idx]["counter"]['O'] for subcircuit_idx in range(self.num_subcircuits)]
        self.gen_prep_cuts()
        # print(self.prep_cuts)
        self.gen_meas_cuts()
        # print(self.meas_cuts)
        self.accumulate_effective_qubits()
        # print(self.acc_eff_qubits)
        self.generate_output_order()
        # print(self.output_orders)
        # print(self.permute_orders)
        self.read_all_probs(prior=prior)

    def add_coef_folder(self, folder_name):
        self.coef_folder = folder_name
    
    def gen_misc_infos(self):
        temp = 0
        num_entries = []
        for subcircuit_idx in range(self.num_subcircuits):
            num_entries.append(len(list(self.entry_dict[subcircuit_idx])))
            temp += len(list(self.entry_dict[subcircuit_idx]))
        self.num_total_entries = temp
        self.num_entries = num_entries
        subcircuit_num_qubits = []
        for subcircuit_idx in range(self.num_subcircuits):
            subcircuit_num_qubits.append(self.subcircuits_info[subcircuit_idx]['num_qubits'])
        self.subcircuit_num_qubits = subcircuit_num_qubits
        
    def read_all_probs(self, prior):
        if self.verbose:
            print("-----> Reading data")
        entry_probs = []
        for subcircuit_idx in range(self.num_subcircuits):
            subcircuit_entry_probs = []
            for entry_idx in range(len(list(self.entry_dict[subcircuit_idx]))):
                entry_count = torch.load(f'{self.data_folder}/subcircuit_{subcircuit_idx}_entry_{entry_idx}.pt', weights_only=True)
                qiskit_permute_order = [i for i in range(self.subcircuit_num_qubits[subcircuit_idx])][::-1]
                entry_count = entry_count.permute(tuple(qiskit_permute_order))
                subcircuit_entry_probs.append(entry_count / torch.sum(entry_count))
            entry_probs.append(torch.stack(subcircuit_entry_probs).to(self.device))
        self.entry_probs = entry_probs

    def load_probs(self, subcircuit_idx, entry_range, prior):
        return

    def gen_prep_cuts(self):
        prep_cuts = []
        prep_cuts_loc = []
        for subcircuit_idx in range(self.num_subcircuits):
            subcircuit_prep_cuts = []
            subcircuit_prep_cuts_loc = []
            for p_cut_loc in self.subcircuits_info[subcircuit_idx]['p_cuts']:
                subcircuit_prep_cuts.append(self.info["cut_index"][self.subcircuits_info[subcircuit_idx]['p_cuts'][p_cut_loc]])
                subcircuit_prep_cuts_loc.append(p_cut_loc)
            prep_cuts.append(subcircuit_prep_cuts)
            prep_cuts_loc.append(subcircuit_prep_cuts_loc)
        self.prep_cuts = prep_cuts
        self.prep_cuts_loc = prep_cuts_loc

    def gen_meas_cuts(self):
        meas_cuts = []
        meas_cuts_loc = []
        for subcircuit_idx in range(self.num_subcircuits):
            subcircuit_meas_cuts = []
            subcircuit_meas_cuts_loc = []
            for m_cut_loc in self.subcircuits_info[subcircuit_idx]['m_cuts']:
                subcircuit_meas_cuts.append(self.info["cut_index"][self.subcircuits_info[subcircuit_idx]['m_cuts'][m_cut_loc]])
                subcircuit_meas_cuts_loc.append(m_cut_loc)
            meas_cuts.append(subcircuit_meas_cuts)
            meas_cuts_loc.append(subcircuit_meas_cuts_loc)
        self.meas_cuts = meas_cuts
        self.meas_cuts_loc = meas_cuts_loc

    def accumulate_effective_qubits(self):
        acc_eff_qubits = [0]
        acc = 0
        for subcircuit_idx in range(self.num_subcircuits):
            acc += self.subcircuits_info[subcircuit_idx]['counter']['effective']
            acc_eff_qubits.append(acc)
        self.acc_eff_qubits = acc_eff_qubits

    def generate_output_order(self):
        output_orders = []
        permute_orders = []
        for subcircuit_idx in range(self.num_subcircuits):
            subcircuit_order = [-1 for _ in range(self.subcircuit_num_qubits[subcircuit_idx])]
            permute_order = []
            count = 0
            for qubit in range(self.subcircuit_num_qubits[subcircuit_idx]):
                if qubit in self.meas_cuts_loc[subcircuit_idx]:
                    continue
                else:
                    subcircuit_order[qubit] = count
                    count += 1
                    permute_order.append(qubit)
            for meas_cut in self.meas_cuts_loc[subcircuit_idx]:
                subcircuit_order[meas_cut] = count
                count += 1
                permute_order.append(meas_cut)
            output_orders.append(torch.tensor(subcircuit_order).to(self.device))
            permute_orders.append(torch.tensor(permute_order).to(self.device))
        self.permute_orders = permute_orders
        self.output_orders = output_orders
