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
    if os.path.exists(data_folder):
        subprocess.run(["rm", "-r", data_folder])
    os.makedirs(data_folder)
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        for job in range(len(subcircuits_entries[subcircuit_idx])):
            if not os.path.exists("%s/subcircuit_%d_entry_%d" % (data_folder, subcircuit_idx, job)):
                counts = {}
                for bitstring in itertools.product('01', repeat = subcircuit.num_qubits):
                    counts[''.join(bitstring)] = 0
                if job == 0:
                    print(counts)
                pickle.dump(
                    {
                        "total_shots": 0,
                        "counts": counts,
                    },
                    open("%s/subcircuit_%d_entry_%d.pckl" % (data_folder, subcircuit_idx, job), "wb"),
                )