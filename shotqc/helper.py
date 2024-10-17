import random
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit
import itertools, os, subprocess, time, torch

def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest

def find_process_jobs(jobs, rank, num_workers):
    count = int(len(jobs) / num_workers)
    remainder = len(jobs) % num_workers
    if rank < remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs

def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if vertex.op.name != "qpd_1q":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def initialize_counts(data_folder, subcircuits, subcircuits_entries):
    if os.path.exists(data_folder):
        subprocess.run(["rm", "-r", data_folder])
    os.makedirs(data_folder)
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        for entry_idx in range(len(subcircuits_entries[subcircuit_idx])):
            count = torch.zeros((2,)*subcircuit.num_qubits)
            torch.save(count, f'{data_folder}/subcircuit_{subcircuit_idx}_entry_{entry_idx}.pt')

def generate_prob_from_counts_with_prior(counts, prior=0):
    total_num = counts["total_shots"]
    prob = {}
    num_bits = len(list(counts["counts"].keys())[0])
    for key in counts["counts"].keys():
        prob[key] = (counts["counts"][key]+prior) / (total_num + 2**num_bits * prior)
    return prob

def params_list_to_matrix(params_list, prep_states):
    if prep_states == [0,2,4,5]:
        assert len(params_list) % 8 == 0, "Illegal list length"
        return params_list.view(-1, 8)
    elif prep_states == range(6):
        assert len(params_list) % 24 == 0, "Illegal list length"
        return params_list.view(-1, 24)
    else:
        raise Exception("current state set not supported")

# def read_probs_with_prior(data_folder, prior):
#     meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
#     entry_dict = meta_info["entry_dict"]
#     subcircuits = meta_info["subcircuits"]
#     prob_with_prior = []
#     for subcircuit_idx in range(len(subcircuits)):
#         subcircuit_entries = []
#         for entry_idx in range(len(list(entry_dict[subcircuit_idx].keys()))):
#             counts = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, entry_idx), "rb"))
#             probs = generate_prob_from_counts_with_prior(counts, prior)
#             subcircuit_entries.append(probs)
#         prob_with_prior.append(subcircuit_entries)
#     return prob_with_prior

def params_matrix_to_list(params):
    return torch.flatten(params)

# def generate_matrix_old(params, prep_states):
#     M = []
#     for param in params:
#         M_cut = []
#         if prep_states == [0, 2, 4, 5]:
#             assert len(param) == 8
#             # param = [a1, a3, a5, a6, b1, b3, b5, b6]
#             M_cut = [
#                 [2-param[0], -2-param[0], -param[4], -param[4], param[0]+param[4], param[0]+param[4]],
#                 [0 for _ in range(6)],
#                 [-param[1], -param[1], 2-param[5], -2-param[5], param[1]+param[5], param[1]+param[5]],
#                 [0 for _ in range(6)],
#                 [-1-param[2], 1-param[2], -1-param[6], 1-param[6], 2+param[2]+param[6], param[2]+param[6]],
#                 [-1-param[3], 1-param[3], -1-param[7], 1-param[7], param[3]+param[7], 2+param[3]+param[7]]
#             ]
#         elif prep_states == range(6):
#             assert len(param) == 24
#             a = param[:6]
#             b = param[6:12]
#             c = param[12:18]
#             d = param[18:]
#             M_cut = [
#                 [1-a[0]-c[0], -1-a[0]-c[1], -b[0]-c[2], -b[0]-c[3], a[0]+b[0]-c[4], a[0]+b[0]-c[5]],
#                 [-1-a[1]-c[0], 1-a[1]-c[1], -b[1]-c[2], -b[1]-c[3], a[1]+b[1]-c[4], a[1]+b[1]-c[5]],
#                 [-a[2]-d[0], -a[2]-d[1], 1-b[2]-d[2], -1-b[2]-d[3], a[2]+b[2]-d[4], a[2]+b[2]-d[5]],
#                 [-a[3]-d[0], -a[3]-d[1], -1-b[3]-d[2], 1-b[3]-d[3], a[3]+b[3]-d[4], a[3]+b[3]-d[5]],
#                 [-a[4]+c[0]+d[0], -a[4]+c[1]+d[1], -b[4]+c[2]+d[2], -b[4]+c[3]+d[3], 2+a[4]+b[4]+c[4]+d[4], a[4]+b[4]+c[5]+d[5]],
#                 [-a[5]+c[0]+d[0], -a[5]+c[1]+d[1], -b[5]+c[2]+d[2], -b[5]+c[3]+d[3], a[5]+b[5]+c[4]+d[4], 2+a[5]+b[5]+c[5]+d[5]]
#             ]
#         else:
#             raise Exception("current state set not supported")
#         M_cut = [[M_cut[i][j]/2 for j in range(6)] for i in range(6)]
#         M.append(M_cut)
#     return M

def generate_matrix(params, prep_states):
    # Stack all tensors in params into a single tensor
    if prep_states == [0, 2, 4, 5]:
        assert params.shape[1] == 8, "params must have shape (batch_size, 8) when prep_states = [0, 2, 4, 5]"
        # Unpack the params tensor
        a1, a3, a5, a6, b1, b3, b5, b6 = params.T  # Unpack columns

        # Create the 6x6 matrices for each entry in params
        M = torch.stack([
            torch.stack([2 - a1, -2 - a1, -b1, -b1, a1 + b1, a1 + b1], dim=-1),
            torch.zeros_like(a1).unsqueeze(-1).repeat(1, 6),  # Vector of zeros
            torch.stack([-a3, -a3, 2 - b3, -2 - b3, a3 + b3, a3 + b3], dim=-1),
            torch.zeros_like(a3).unsqueeze(-1).repeat(1, 6),
            torch.stack([-1 - a5, 1 - a5, -1 - b5, 1 - b5, 2 + a5 + b5, a5 + b5], dim=-1),
            torch.stack([-1 - a6, 1 - a6, -1 - b6, 1 - b6, a6 + b6, 2 + a6 + b6], dim=-1)
        ], dim=1)  # Stack along the second dimension to form the 6x6 matrix for each element

    elif prep_states == range(6):
        assert params.shape[1] == 24, "params must have shape (batch_size, 24) when prep_states = range(6)"
        # Unpack the params tensor
        a = params[:, :6]
        b = params[:, 6:12]
        c = params[:, 12:18]
        d = params[:, 18:]

        # Create the 6x6 matrices for each entry in params
        M = torch.stack([
            torch.stack([1 - a[:, 0] - c[:, 0], -1 - a[:, 0] - c[:, 1], -b[:, 0] - c[:, 2], -b[:, 0] - c[:, 3], a[:, 0] + b[:, 0] - c[:, 4], a[:, 0] + b[:, 0] - c[:, 5]], dim=-1),
            torch.stack([-1 - a[:, 1] - c[:, 0], 1 - a[:, 1] - c[:, 1], -b[:, 1] - c[:, 2], -b[:, 1] - c[:, 3], a[:, 1] + b[:, 1] - c[:, 4], a[:, 1] + b[:, 1] - c[:, 5]], dim=-1),
            torch.stack([-a[:, 2] - d[:, 0], -a[:, 2] - d[:, 1], 1 - b[:, 2] - d[:, 2], -1 - b[:, 2] - d[:, 3], a[:, 2] + b[:, 2] - d[:, 4], a[:, 2] + b[:, 2] - d[:, 5]], dim=-1),
            torch.stack([-a[:, 3] - d[:, 0], -a[:, 3] - d[:, 1], -1 - b[:, 3] - d[:, 2], 1 - b[:, 3] - d[:, 3], a[:, 3] + b[:, 3] - d[:, 4], a[:, 3] + b[:, 3] - d[:, 5]], dim=-1),
            torch.stack([-a[:, 4] + c[:, 0] + d[:, 0], -a[:, 4] + c[:, 1] + d[:, 1], -b[:, 4] + c[:, 2] + d[:, 2], -b[:, 4] + c[:, 3] + d[:, 3], 2 + a[:, 4] + b[:, 4] + c[:, 4] + d[:, 4], a[:, 4] + b[:, 4] + c[:, 5] + d[:, 5]], dim=-1),
            torch.stack([-a[:, 5] + c[:, 0] + d[:, 0], -a[:, 5] + c[:, 1] + d[:, 1], -b[:, 5] + c[:, 2] + d[:, 2], -b[:, 5] + c[:, 3] + d[:, 3], a[:, 5] + b[:, 5] + c[:, 4] + d[:, 4], 2 + a[:, 5] + b[:, 5] + c[:, 5] + d[:, 5]], dim=-1)
        ], dim=1)

    else:
        raise Exception("Current state set not supported")

    M = M / 2  # Element-wise division by 2, preserving gradient
    return M

def tensor_product(tensors, device):
    n = len(tensors)
    if n == 0:
        return torch.tensor(1., device=device)
    tensor_size = tensors[0].shape[0]
    indices = torch.ones((n,n), dtype=torch.int)
    for i in range(n):
        indices[i][i] = tensor_size
    product = tensors[0].view(tuple(indices[0].tolist()))
    for i in range(n-1):
        temp = tensors[i+1].view(tuple(indices[i+1].tolist()))
        product = product * temp
    return product

def find_slices(bitstrings, m_cuts_loc):
    all_slices = []
    for bitstring in bitstrings:
        bit_idx = 0
        slices = []
        for i in range(len(bitstring) + len(m_cuts_loc)):
            if i in m_cuts_loc:
                slices.append(slice(None))
            else:
                slices.append(bitstring[bit_idx])
                bit_idx += 1
        all_slices.append(tuple(slices))
    return all_slices

def generate_all_bitstrings(n):
    # Number of bitstrings: 2^n
    num_bitstrings = 2 ** n
    
    # Generate integers from 0 to 2^n - 1
    integers = torch.arange(num_bitstrings, dtype=torch.int64)
    
    # Convert integers to binary (bitstrings) and represent as an n-bit tensor
    bitstrings = ((integers.unsqueeze(1) >> torch.arange(n-1, -1, -1)) & 1).to(torch.int64)
    
    return bitstrings

def generate_relative_bitstrings(bitstrings, num_meas, order_list):
    batch_size = bitstrings.shape[0]
    bit_len = bitstrings.shape[1]
    new_bitstrings = bitstrings.view((batch_size//(2**num_meas), 2**num_meas, bit_len))
    permuted_bitstrings = new_bitstrings[...,order_list]
    return permuted_bitstrings

def bitstring_batch_generator(num_qubits, batch_size):
    # Generate bitstrings on the fly and yield them in batches
    bitstring_iterator = itertools.product([0, 1], repeat=num_qubits)
    
    while True:
        batch = list(itertools.islice(bitstring_iterator, batch_size))
        if not batch:
            break
        yield torch.tensor(batch, dtype=torch.int64)

def advanced_indexing(tensor, indexes):
    indices = tuple(indexes[:, i] for i in range(indexes.shape[1]))
    result = tensor[indices]
    return result