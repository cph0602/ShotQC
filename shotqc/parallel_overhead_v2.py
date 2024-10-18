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


# def calc_subcircuit_value(coef_matrix, args, batch, prep_config, subcircuit_idx, device):
#     # calculate coef tensor product
#     #  *. Misc Variables
#     num_meas = args.num_meas[subcircuit_idx]
#     batch_size = batch.shape[0]
#     #  0. Find base local entry idx: entries from base -> base + 3^num_meas -1 are used
#     base_local_entry_idx = 0
#     for prep_cut in args.prep_cuts[subcircuit_idx]:
#         base_local_entry_idx *= args.len_prep_states
#         base_local_entry_idx += prep_config[prep_cut]
#     entry_batch_idx = base_local_entry_idx
#     base_local_entry_idx *= 3**num_meas
#     #  1. fetch coef matrix lines that correspond to measurement cut indexes
#     coef_matrix_rows = []
#     for meas_cut in args.meas_cuts[subcircuit_idx]:
#         coef_matrix_rows.append(coef_matrix[meas_cut][args.prep_states[prep_config[meas_cut]]][:])
#     #  2. perform tensor product
#     # moved back
#     #  3. find correct section of entry_probs that correspond to this prep_config
#     subcircuit_bitstrings = torch.flip(batch, dims=[1])[:, args.acc_eff_qubits[subcircuit_idx]:args.acc_eff_qubits[subcircuit_idx+1]]
#     indices = tuple(subcircuit_bitstrings[:, i] for i in range(subcircuit_bitstrings.shape[1]))
#     padded_permute_order = torch.cat((torch.tensor([0],device=device), args.permute_orders[subcircuit_idx] + torch.ones_like(args.permute_orders[subcircuit_idx], device=device)))
#     permuted_probs = torch.permute(args.entry_probs[subcircuit_idx][base_local_entry_idx:base_local_entry_idx + 3**num_meas], tuple(padded_permute_order)).contiguous()
#     # print(permuted_probs.shape)
#     final_permute_order = (1,0) + tuple([i+2 for i in range(num_meas)])
#     used_probs = permuted_probs[(slice(None),) + indices].permute(final_permute_order).contiguous()
#     #  4. perform element-wise multiplication
#     #    (a) lining them up
#     coef_permute_tuple = tuple([i*2 for i in range(num_meas)]) + tuple([i*2+1 for i in range(num_meas)])
#     coef_tensor_product = tensor_product(coef_matrix_rows, device) # 6^(#meas)
#     lined_coef_products = coef_tensor_product.view((3,2)*num_meas).permute(coef_permute_tuple).contiguous().view((3**num_meas,)+(2,)*num_meas)
#     #    (b) multiply
#     #  4.5. use sum() for subcircuit value
#     sum_dimensions = tuple([i+1 for i in range(num_meas+1)])
#     subcircuit_value = torch.sum(used_probs * lined_coef_products, dim=(sum_dimensions))
#     print(lined_coef_products.shape)
#     #  5. return result, and remember to tell main function which section it is
#     # config_prob_coef = torch.zeros((batch_size,)+args.entry_probs[subcircuit_idx].shape, device=device)
#     # index: 
#     output_tensor = torch.zeros((subcircuit_bitstrings.shape[0],) + (2,)*subcircuit_bitstrings.shape[1], device=device)
#     batch_indices = torch.arange(subcircuit_bitstrings.shape[0])
#     output_tensor[(batch_indices,) + tuple(subcircuit_bitstrings.t())] = 1
#     output_view = (subcircuit_bitstrings.shape[0],1) + (2,)*subcircuit_bitstrings.shape[1] + (1,)*num_meas
#     output_repeat = (1,3**num_meas) + (1,)*subcircuit_bitstrings.shape[1] + (2,)*num_meas
#     output_tensor = output_tensor.view(output_view).repeat(output_repeat)
#     repeated_coefs = lined_coef_products.view(output_repeat).repeat(output_view)
#     final_product = output_tensor * repeated_coefs
#     print(final_product.shape)
#     # config_prob_coef[:, base_local_entry_idx:base_local_entry_idx + 3**num_meas] = final_product
#     # print(config_prob_coef[1,0,0,0,0,0,0])
#     padded_unpermute_order = torch.cat((torch.tensor([0, 1],device=device), args.output_orders[subcircuit_idx] + 2 * torch.ones_like(args.output_orders[subcircuit_idx], device=device)))
#     # print(padded_unpermute_order)
#     return subcircuit_value, torch.permute(final_product, tuple(padded_unpermute_order)), entry_batch_idx

# def save_prob_coef(args, subcircuit_idx, entry_batch_idx, prob_coef):
#     if os.path.exists(f'{args.coef_folder}/subcircuit_{subcircuit_idx}_prep_{entry_batch_idx}_coef.pt'):
#         prev_coef = torch.load(f'{args.coef_folder}/subcircuit_{subcircuit_idx}_prep_{entry_batch_idx}_coef.pt', weights_only=True)
#         torch.save(prev_coef + prob_coef, f'{args.coef_folder}/subcircuit_{subcircuit_idx}_prep_{entry_batch_idx}_coef.pt')
#         del prev_coef
#         del prob_coef
#     else:
#         torch.save(prob_coef, f'{args.coef_folder}/subcircuit_{subcircuit_idx}_prep_{entry_batch_idx}_coef.pt')
#         del prob_coef

# def load_prob_coef(args, subcircuit_idx, entry_idx):
#     entry_batch_idx = entry_idx // (3 ** args.num_meas[subcircuit_idx])
#     inner_idx = entry_idx % (3 ** args.num_meas[subcircuit_idx])
#     prob_coef = torch.load(f'{args.coef_folder}/subcircuit_{subcircuit_idx}_prep_{entry_batch_idx}_coef.pt', weights_only=True)
#     return prob_coef[:, inner_idx]

# def total_entry_coef(params, args, batch_size, device=None):
#     # Handle in batches to avoid memory overload
#     ## hyperparameters
#     # batch_size = 1024
#     # start_time=perf_counter()
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ## main
#     params = params.to(device)
#     params_matrix = params_list_to_matrix(params, args.prep_states)
#     coef_matrix = generate_matrix(params_matrix, args.prep_states)
#     entry_coef = [torch.zeros(args.num_entries[subcircuit_idx], device=device) for subcircuit_idx in range(args.num_subcircuits)]
#     total_values = torch.tensor([], device=device)
#     num_batch = ceil(2**args.num_qubits/batch_size)
#     batch_cnt = 0
#     for batch in bitstring_batch_generator(args.num_qubits, batch_size):
#         start_time = perf_counter()
#         if os.path.exists(f'{args.data_folder}/batch_coef_tmp'):
#             subprocess.run(["rm", "-r", f'{args.data_folder}/batch_coef_tmp'])
#         os.makedirs(f'{args.data_folder}/batch_coef_tmp')
#         args.add_coef_folder(f'{args.data_folder}/batch_coef_tmp')
#         # Calculate prob_coef
#         this_batch_size = batch.shape[0]
#         this_total_value = torch.zeros(this_batch_size, device=device)
#         # prob_coef = [
#         #     torch.zeros((this_batch_size,) + args.entry_probs[subcircuit_idx].shape, 
#         #     requires_grad=True, device=device)
#         #     for subcircuit_idx in range(args.num_subcircuits)
#         # ]
#         # print(prob_coef[0].shape)
#         for prep_config in itertools.product(range(args.len_prep_states), repeat=args.num_cuts):
#             save_time = 0
#             prep_start_time = perf_counter()
#             # print("prep_config: ", prep_config)
#             value = torch.ones(this_batch_size, device=device)
#             subcircuit_values = []
#             config_prob_coef = []
#             entry_batch_idxs = []
#             for subcircuit_idx in range(args.num_subcircuits):
#                 # going through a batch of entries
#                 # e.g. for a subcircuit with 1 prep, 2 meas => 36 subcircuits
#                 #      => each time go through a 0-8, 9-17, 18-26, 27-35
#                 subcircuit_value, subcircuit_config_prob_coef, entry_batch_idx = calc_subcircuit_value(
#                     coef_matrix, args, batch, prep_config, subcircuit_idx, device
#                 )
#                 # print(subcircuit_config_prob_coef[1,0,0,0,0,0,0])
#                 subcircuit_values.append(subcircuit_value)
#                 config_prob_coef.append(subcircuit_config_prob_coef)
#                 entry_batch_idxs.append(entry_batch_idx)
#                 value = value * subcircuit_value
#             this_total_value = this_total_value + value
#             subcircuit_values = torch.stack(subcircuit_values)
#             product_except_self = [torch.ones(this_batch_size, device=device) for subcircuit_idx in range(args.num_subcircuits)]
#             for subcircuit_idx in range(args.num_subcircuits):
#                 mask = torch.ones(args.num_subcircuits, dtype=bool)
#                 mask[subcircuit_idx] = False
#                 product_except_self[subcircuit_idx] = torch.prod(subcircuit_values[mask], dim=0)
#                 broadcast_view = (this_batch_size, 1) + (1,)*args.subcircuit_num_qubits[subcircuit_idx]
#                 save_start = perf_counter()
#                 save_prob_coef(args, subcircuit_idx, entry_batch_idxs[subcircuit_idx], product_except_self[subcircuit_idx].view(broadcast_view) * config_prob_coef[subcircuit_idx])
#                 save_time += perf_counter() - save_start
#                 # print(config_prob_coef[subcircuit_idx][1,0,0,0,0,0,0])
#             prep_elapsed_time = perf_counter()-prep_start_time
#             # print("save: ", save_time)
#             # print(f"One prep config took {prep_elapsed_time} seconds. Estimate total {prep_elapsed_time * args.len_prep_states ** args.num_cuts} seconds.")
#         total_values = torch.cat((total_values, this_total_value))
#         # print("value:",prob_coef[0][10,0,0,0,1,0,1])
#         # Calculate variance
#         var_start = perf_counter()
#         for subcircuit_idx in range(args.num_subcircuits):
#             num_qubit = args.subcircuit_num_qubits[subcircuit_idx]
#             num_meas = args.num_meas[subcircuit_idx]
#             for entry_idx in range(args.num_entries[subcircuit_idx]):
#                 # 0. Fetch probs & coefs & gen bitstrings
#                 entry_probs = args.entry_probs[subcircuit_idx][entry_idx] # shape=(batch_size, 2,2,...)
#                 entry_prob_coefs = load_prob_coef(args, subcircuit_idx, entry_idx)
#                 bitstrings = generate_all_bitstrings(num_qubit).to(device)
#                 # 1. Self variance
#                 self_probs = entry_probs[tuple(bitstrings.t())]
#                 self_coefs = entry_prob_coefs[(slice(None),)+tuple(bitstrings.t())] # shape=(batch_size,2**num_qubits)
#                 one_minus_self_probs = torch.ones_like(self_probs, requires_grad=True) - self_probs
#                 sum_variance = torch.sum(self_probs * one_minus_self_probs * self_coefs * self_coefs, dim=(1))
#                 # if subcircuit_idx == 0 and entry_idx == 4:
#                 #     print(sum_variance[11]) ## checked: variance correct
#                 # 2. Covariance
#                 #   (a) Find all relative bitstrings
#                 reordered_bitstrings = bitstrings[..., args.output_orders[subcircuit_idx]]
#                 # print(reordered_bitstrings)
#                 #   (b) calculate covariance
#                 # print(subcircuit_idx, entry_idx)
#                 # print("Entry_prob_coef", entry_prob_coefs)

#                 coefs = entry_prob_coefs[(slice(None),)+tuple(reordered_bitstrings.t())].view((this_batch_size, bitstrings.shape[0]//(2**num_meas), 2**num_meas))
#                 probs = entry_probs[tuple(reordered_bitstrings.t())].view((bitstrings.shape[0]//(2**num_meas), 2**num_meas))
#                 outer_coef = torch.einsum('bij,bik->bijk', coefs, coefs)
#                 outer_prob = torch.einsum('bi,bj->bij', probs, probs)
#                 # print(outer_coef.shape)
#                 # print(outer_prob.shape)
#                 # outer_coef = outer_coef.view(this_batch_size, bitstrings.shape[0]//(2**num_meas), -1)
#                 # outer_prob = outer_prob.view(bitstrings.shape[0]//(2**num_meas), -1)
#                 # print((outer_coef * outer_prob).shape)
#                 covariance = torch.sum(self_coefs * self_coefs * self_probs * self_probs, dim=(1)) - torch.sum(outer_coef * outer_prob, dim=(1,2,3))
#                 # print(covariance)
#                 entry_coef[subcircuit_idx][entry_idx] = torch.sum(sum_variance + covariance)
#         # print(f"variance calculations took {perf_counter()-var_start} seconds")
#         # print(f"loading and saving tensors took {save_time} seconds")
#                 # if subcircuit_idx == 0 and entry_idx == 4:
#                 #     print(sum_variance[11]+covariance[11])
#         #         break
#         #     break
#     # print("Total time: ", perf_counter()-start_time)
#         # print(f"-----> Batch {batch_cnt+1}/{num_batch} Completed, used {perf_counter()-start_time} seconds")
#     return entry_coef

# def parallel_cost_function(params, args, device=None, batch_size=1024):
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     entry_coef = total_entry_coef(params, args, batch_size)
#     cost = torch.tensor(0., requires_grad=True, device=device)
#     for subcircuit_idx in range(args.num_subcircuits):
#         cost = cost + torch.sum(torch.sqrt(entry_coef[subcircuit_idx]))
#     return cost


# # currently using grad #
# def parallel_distribute(params, args, total_samples, device=None, batch_size=1024):
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     with torch.no_grad():
#         entry_coef = total_entry_coef_grad(params, args, batch_size)
#         total_cost = parallel_cost_function_grad(params, args, device, batch_size)
#     distributions = []
#     sample = torch.tensor(total_samples, device=device)
#     for subcircuit_idx in range(args.num_subcircuits):
#         subcircuit_distribution = torch.floor(sample * torch.sqrt(entry_coef[subcircuit_idx]) / total_cost)
#         distributions.append(subcircuit_distribution.tolist())
#     # print(distributions)
#     return distributions


# ################ GRAD VERSIONS FOR SGD ################
            
# def calc_subcircuit_value_grad(coef_matrix, args, batch, prep_config, subcircuit_idx, device):
#     # calculate coef tensor product with gradient
#     #  *. Misc Variables
#     num_meas = args.num_meas[subcircuit_idx]
#     batch_size = batch.shape[0]
#     #  0. Find base local entry idx: entries from base -> base + 3^num_meas -1 are used
#     base_local_entry_idx = 0
#     for prep_cut in args.prep_cuts[subcircuit_idx]:
#         base_local_entry_idx *= args.len_prep_states
#         base_local_entry_idx += prep_config[prep_cut]
#     base_local_entry_idx *= 3**num_meas
#     #  1. fetch coef matrix lines that correspond to measurement cut indexes
#     coef_matrix_rows = []
#     for meas_cut in args.meas_cuts[subcircuit_idx]:
#         coef_matrix_rows.append(coef_matrix[meas_cut][args.prep_states[prep_config[meas_cut]]][:])
#     #  2. perform tensor product
#     # moved back
#     #  3. find correct section of entry_probs that correspond to this prep_config
#     subcircuit_bitstrings = torch.flip(batch, dims=[1])[:, args.acc_eff_qubits[subcircuit_idx]:args.acc_eff_qubits[subcircuit_idx+1]]
#     indices = tuple(subcircuit_bitstrings[:, i] for i in range(subcircuit_bitstrings.shape[1]))
#     padded_permute_order = torch.cat((torch.tensor([0],device=device), args.permute_orders[subcircuit_idx] + torch.ones_like(args.permute_orders[subcircuit_idx], device=device)))
#     permuted_probs = torch.permute(args.entry_probs[subcircuit_idx][base_local_entry_idx:base_local_entry_idx + 3**num_meas], tuple(padded_permute_order)).contiguous()
#     final_permute_order = (1,0) + tuple([i+2 for i in range(num_meas)])
#     used_probs = permuted_probs[(slice(None),) + indices].permute(final_permute_order).contiguous()
#     #  4. perform element-wise multiplication
#     #    (a) lining them up
#     coef_permute_tuple = tuple([i*2 for i in range(num_meas)]) + tuple([i*2+1 for i in range(num_meas)])
#     coef_tensor_product = tensor_product(coef_matrix_rows, device)
#     lined_coef_products = coef_tensor_product.view((3,2)*num_meas).permute(coef_permute_tuple).contiguous().view((3**num_meas,)+(2,)*num_meas)
#     #    (b) multiply
#     #  4.5. use sum() for subcircuit value
#     sum_dimensions = tuple([i+1 for i in range(num_meas+1)])
#     subcircuit_value = torch.sum(used_probs * lined_coef_products, dim=(sum_dimensions))
#     #  5. return result, and remember to tell main function which section it is
#     config_prob_coef = torch.zeros((batch_size,)+args.entry_probs[subcircuit_idx].shape, device=device)
    
#     output_tensor = torch.zeros((subcircuit_bitstrings.shape[0],) + (2,)*subcircuit_bitstrings.shape[1], device=device)
#     batch_indices = torch.arange(subcircuit_bitstrings.shape[0])
#     output_tensor[(batch_indices,) + tuple(subcircuit_bitstrings.t())] = 1
#     output_view = (subcircuit_bitstrings.shape[0],1) + (2,)*subcircuit_bitstrings.shape[1] + (1,)*num_meas
#     output_repeat = (1,3**num_meas) + (1,)*subcircuit_bitstrings.shape[1] + (2,)*num_meas
#     output_tensor = output_tensor.view(output_view).repeat(output_repeat)
#     repeated_coefs = lined_coef_products.view(output_repeat).repeat(output_view)
#     final_product = output_tensor * repeated_coefs
#     config_prob_coef[:, base_local_entry_idx:base_local_entry_idx + 3**num_meas] = final_product
#     padded_unpermute_order = torch.cat((torch.tensor([0, 1],device=device), args.output_orders[subcircuit_idx] + 2 * torch.ones_like(args.output_orders[subcircuit_idx], device=device)))
#     config_prob_coef = torch.permute(config_prob_coef, tuple(padded_unpermute_order))
#     return subcircuit_value, config_prob_coef

# def total_entry_coef_grad(params, args, batch_size, device=None):
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ## main
#     params = params.to(device)
#     params_matrix = params_list_to_matrix(params, args.prep_states)
#     coef_matrix = generate_matrix(params_matrix, args.prep_states)
#     entry_coef = [torch.zeros(args.num_entries[subcircuit_idx], device=device) for subcircuit_idx in range(args.num_subcircuits)]
#     total_values = torch.tensor([], device=device)
#     for batch in bitstring_batch_generator(args.num_qubits, batch_size):
#         # Calculate prob_coef
#         this_batch_size = batch.shape[0]
#         this_total_value = torch.zeros(this_batch_size, device=device)
#         prob_coef = [
#             torch.zeros((this_batch_size,) + args.entry_probs[subcircuit_idx].shape, 
#             requires_grad=True, device=device)
#             for subcircuit_idx in range(args.num_subcircuits)
#         ]
#         for prep_config in itertools.product(range(args.len_prep_states), repeat=args.num_cuts):
#             value = torch.ones(this_batch_size, device=device)
#             subcircuit_values = []
#             config_prob_coef = []
#             for subcircuit_idx in range(args.num_subcircuits):
#                 # going through a batch of entries
#                 subcircuit_value, subcircuit_config_prob_coef = calc_subcircuit_value_grad(
#                     coef_matrix, args, batch, prep_config, subcircuit_idx, device
#                 )
#                 subcircuit_values.append(subcircuit_value)
#                 config_prob_coef.append(subcircuit_config_prob_coef)
#                 value = value * subcircuit_value
#             this_total_value = this_total_value + value
#             subcircuit_values = torch.stack(subcircuit_values)
#             product_except_self = [torch.ones(this_batch_size, device=device) for subcircuit_idx in range(args.num_subcircuits)]
#             for subcircuit_idx in range(args.num_subcircuits):
#                 mask = torch.ones(args.num_subcircuits, dtype=bool)
#                 mask[subcircuit_idx] = False
#                 product_except_self[subcircuit_idx] = torch.prod(subcircuit_values[mask], dim=0)
#                 broadcast_view = (this_batch_size, 1) + (1,)*args.subcircuit_num_qubits[subcircuit_idx]
#                 prob_coef[subcircuit_idx] = prob_coef[subcircuit_idx] + product_except_self[subcircuit_idx].view(broadcast_view) * config_prob_coef[subcircuit_idx]
#         total_values = torch.cat((total_values, this_total_value))
#         # Calculate variance
#         for subcircuit_idx in range(args.num_subcircuits):
#             num_qubit = args.subcircuit_num_qubits[subcircuit_idx]
#             num_meas = args.num_meas[subcircuit_idx]
#             for entry_idx in range(args.num_entries[subcircuit_idx]):
#                 # 0. Fetch probs & coefs & gen bitstrings
#                 entry_probs = args.entry_probs[subcircuit_idx][entry_idx]
#                 entry_prob_coefs = prob_coef[subcircuit_idx][:,entry_idx] # shape=(batch_size, 2,2,...)
#                 bitstrings = generate_all_bitstrings(num_qubit).to(device)
#                 # 1. Self variance
#                 self_probs = entry_probs[tuple(bitstrings.t())]
#                 self_coefs = entry_prob_coefs[(slice(None),)+tuple(bitstrings.t())] # shape=(batch_size,2**num_qubits)
#                 one_minus_self_probs = torch.ones_like(self_probs, requires_grad=True) - self_probs
#                 sum_variance = torch.sum(self_probs * one_minus_self_probs * self_coefs * self_coefs, dim=(1))
#                 # 2. Covariance
#                 #   (a) Find all relative bitstrings
#                 reordered_bitstrings = bitstrings[..., args.output_orders[subcircuit_idx]]
#                 coefs = entry_prob_coefs[(slice(None),)+tuple(reordered_bitstrings.t())].view((this_batch_size, bitstrings.shape[0]//(2**num_meas), 2**num_meas))
#                 probs = entry_probs[tuple(reordered_bitstrings.t())].view((bitstrings.shape[0]//(2**num_meas), 2**num_meas))
#                 outer_coef = torch.einsum('bij,bik->bijk', coefs, coefs)
#                 outer_prob = torch.einsum('bi,bj->bij', probs, probs)
#                 covariance = torch.sum(self_coefs * self_coefs * self_probs * self_probs, dim=(1)) - torch.sum(outer_coef * outer_prob, dim=(1,2,3))
#                 entry_coef[subcircuit_idx][entry_idx] = entry_coef[subcircuit_idx][entry_idx] + torch.sum(sum_variance + covariance)
#     return entry_coef

# def parallel_cost_function_grad(params, args, device=None, batch_size=1024):
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     entry_coef = total_entry_coef_grad(params, args, batch_size)
#     cost = torch.tensor(0., requires_grad=True, device=device)
#     for subcircuit_idx in range(args.num_subcircuits):
#         cost = cost + torch.sum(torch.sqrt(entry_coef[subcircuit_idx]))
#     return cost

############### TEST VERSIONS ################
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

def total_entry_coef(params, args, batch_size, device=None):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## main
    params = params.to(device)
    params_matrix = params_list_to_matrix(params, args.prep_states)
    coef_matrix = generate_matrix(params_matrix, args.prep_states)
    entry_coef = [torch.zeros(args.num_entries[subcircuit_idx], device=device) for subcircuit_idx in range(args.num_subcircuits)]
    # total_values = torch.tensor([], device=device) # for reconstructing
    batch_cnt = 0
    num_batch = ceil(2**args.num_qubits/batch_size)
    for batch in bitstring_batch_generator(args.num_qubits, batch_size):
        start_time = perf_counter()
        # Calculate prob_coef
        this_batch_size = batch.shape[0]
        this_total_value = torch.zeros(this_batch_size, device=device)
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
            this_total_value = this_total_value + value
            subcircuit_values = torch.stack(subcircuit_values)
            product_except_self = [torch.ones(this_batch_size, device=device) for subcircuit_idx in range(args.num_subcircuits)]
            for subcircuit_idx in range(args.num_subcircuits):
                mask = torch.ones(args.num_subcircuits, dtype=bool)
                mask[subcircuit_idx] = False
                product_except_self[subcircuit_idx] = torch.prod(subcircuit_values[mask], dim=0) #(batch_size)
                # print(product_except_self[subcircuit_idx].shape)
                # print(config_prob_coef[subcircuit_idx].shape)
                base_local_entry_idx = 0
                for prep_cut in args.prep_cuts[subcircuit_idx]:
                    base_local_entry_idx *= args.len_prep_states
                    base_local_entry_idx += prep_config[prep_cut]
                base_local_entry_idx *= 3**args.num_meas[subcircuit_idx]
                container = torch.zeros_like(prob_coef[subcircuit_idx])
                container[:, base_local_entry_idx:base_local_entry_idx+3**args.num_meas[subcircuit_idx]] = torch.outer(product_except_self[subcircuit_idx], config_prob_coef[subcircuit_idx].flatten()).reshape((this_batch_size,)+config_prob_coef[subcircuit_idx].shape)
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
                variance = torch.sum(self_probs*one_minus_probs*self_coefs*self_coefs, dim=tuple(range(1,num_meas+1)))
                # 2. Covariance
                #   (a) Find all relative bitstrings
                outer_probs = self_probs.view(this_batch_size, -1, 1) * self_probs.view(this_batch_size, 1, -1)
                outer_coefs = self_coefs.view(this_batch_size, -1, 1) * self_coefs.view(this_batch_size, 1, -1)
                covariance = torch.sum(self_probs*self_probs*self_coefs*self_coefs, dim=tuple(range(1,num_meas+1))) - torch.sum(outer_coefs*outer_probs, dim=(1,2))
                entry_coef[subcircuit_idx][entry_idx] = entry_coef[subcircuit_idx][entry_idx] + torch.sum(variance + covariance)
                del self_coefs, self_probs
                del one_minus_probs
                del outer_coefs, outer_probs
                del variance, covariance
                del subcircuit_bitstrings
                del permuted_probs
        # print(f"-----> Batch {batch_cnt+1}/{num_batch} Completed, used {perf_counter()-start_time} seconds")
        batch_cnt += 1
        del prob_coef
    return entry_coef

def parallel_cost_function(params, args, device=None, batch_size=1024):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entry_coef = total_entry_coef(params, args, batch_size)
    cost = torch.tensor(0., requires_grad=True, device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        cost = cost + torch.sum(torch.sqrt(entry_coef[subcircuit_idx]))
    return cost

def parallel_distribute(params, args, total_samples, device=None, batch_size=1024):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        entry_coef = total_entry_coef(params, args, batch_size)
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

def parallel_variance(params, args, shot_count, device=None, batch_size=1024):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entry_coef = total_entry_coef(params, args, batch_size)
    var = torch.tensor(0., requires_grad=True, device=device)
    for subcircuit_idx in range(args.num_subcircuits):
        temp = torch.tensor(shot_count[subcircuit_idx], device=device)
        var = var + torch.sum(entry_coef[subcircuit_idx] / temp)
    return var

def parallel_reconstruct(params, args, device=None, batch_size=1024, ext_save=False):
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
        print(f"-----> Batch {batch_cnt+1}/{num_batch} Completed, used {perf_counter()-start_time} seconds")
        batch_cnt += 1
    result = {}
    idx = 0
    torch.save(total_values, f'output_tensor.pt')
    for bittuple in itertools.product("01", repeat=args.num_qubits):
        bitstring = ''.join(bittuple)
        result[bitstring] = total_values[idx].item()
        idx += 1
    return result