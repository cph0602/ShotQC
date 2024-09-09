import random
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit
import itertools, pickle, os, subprocess, time

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
        if vertex.op.name != "param_cut":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def initialize_counts(data_folder, subcircuits, subcircuits_entries):
    # if os.path.exists(data_folder):
    #     subprocess.run(["rm", "-r", data_folder])
    # os.makedirs(data_folder)
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        for job in range(len(subcircuits_entries[subcircuit_idx])):
            if not os.path.exists("%s/subcircuit_%d_entry_%d" % (data_folder, subcircuit_idx, job)):
                counts = {}
                for bitstring in itertools.product('01', repeat = subcircuit.num_qubits):
                    counts[''.join(bitstring)] = 0
                pickle.dump(
                    {
                        "total_shots": 0,
                        "counts": counts,
                    },
                    open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, job), "wb"),
                )

def generate_prob_from_counts_with_prior(counts, prior=0):
    total_num = counts["total_shots"]
    prob = {}
    num_bits = len(list(counts["counts"].keys())[0])
    for key in counts["counts"].keys():
        prob[key] = (counts["counts"][key]+prior) / (total_num + 2**num_bits * prior)
    return prob


def params_list_to_matrix(params_list, prep_states):
    params = []
    if prep_states == [0,2,4,5]:
        assert len(params_list) % 8 == 0, "Illegal list length"
        for i in range(len(params_list)//8):
            params.append(params_list[(i*8):(i*8+8)])
    elif prep_states == range(6):
        assert len(params_list) % 24 == 0, "Illegal list length"
        for i in range(len(params_list)//24):
            params.append(params_list[(i*24):(i*24+24)])
    else:
        raise Exception("current state set not supported")
    return params

def read_probs_with_prior(data_folder, prior):
    meta_info = pickle.load(open("%s/meta_info.pckl" % (data_folder), "rb"))
    entry_dict = meta_info["entry_dict"]
    subcircuits = meta_info["subcircuits"]
    prob_with_prior = []
    for subcircuit_idx in range(len(subcircuits)):
        subcircuit_entries = []
        for entry_idx in range(len(list(entry_dict[subcircuit_idx].keys()))):
            counts = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, entry_idx), "rb"))
            probs = generate_prob_from_counts_with_prior(counts, prior)
            subcircuit_entries.append(probs)
        prob_with_prior.append(subcircuit_entries)
    return prob_with_prior

def params_matrix_to_list(params):
    params_list = []
    for row in params:
        for element in row:
            params_list.append(element)
    return params_list

def generate_matrix(params, prep_states):
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
                [-1-param[3], 1-param[3], -1-param[7], 1-param[7], param[3]+param[7], 2+param[3]+param[7]]
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