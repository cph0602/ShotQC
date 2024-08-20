from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from typing import List
import itertools, copy, pickle, subprocess, psutil, os
import numpy as np
from shotqc.helper import find_process_jobs, scrambled, circuit_stripping


def get_num_workers(num_jobs, ram_required_per_worker):
    ram_avail = psutil.virtual_memory().available / 1024**3
    ram_avail = ram_avail / 4 * 3
    num_cpus = int(os.cpu_count() / 4 * 3)
    num_workers = int(min(ram_avail / ram_required_per_worker, num_jobs, num_cpus))
    return num_workers


def run_samples(subcircuits: List[QuantumCircuit], subcircuits_entries: List[List[tuple[List[int]]]], run_mode: str, SD_mode: str, SD_value: List[List[int]] | int, data_folder: str):
    """
    SD_mode = "equal": run each subcircuit variant with SD_value (int) shots each
    SD_mode = "distribute": run each subcircuit variant with its corresponding shot distribution SD_mode[subcircuit_index][subcircuit_entry_index]
    """
    if SD_mode == "equal":
        assert isinstance(SD_value, int)
        SD_dist = [[SD_value for _ in range(len(subcircuits_entries[subcircuit_idx]))] for subcircuit_idx in range(len(subcircuits))]
    elif SD_mode == "distribute":
        SD_dist = copy.deepcopy(SD_value)
    else:
        raise Exception("Invalid SD_mode")
    stripped_subcircuits = [circuit_stripping(subcircuits[i]) for i in range(len(subcircuits))]
    entry_dict = []
    for subcircuit_entries in subcircuits_entries:
        sub_entry = {subcircuit_entries[i]: i for i in range(len(subcircuit_entries))}
        entry_dict.append(sub_entry)
    pickle.dump(
        {
            "subcircuits": stripped_subcircuits,
            "run_mode": run_mode,
            "SD_dist": SD_dist,
            "entry_dict": entry_dict,
        },
        open("%s/meta_info.pckl" % data_folder, "wb"),
    )
    for index, subcircuit_entries in enumerate(subcircuits_entries):
        jobs = range(len(subcircuit_entries))
        num_workers = get_num_workers(
            num_jobs=len(jobs),
            ram_required_per_worker=2 ** subcircuits[index].num_qubits
            * 4
            / 1e9,
        )
        procs = []
        for rank in range(num_workers):
            rank_jobs = find_process_jobs(jobs=jobs, rank=rank, num_workers=num_workers)
            if len(rank_jobs) > 0:
                pickle.dump(
                    rank_jobs, open("%s/rank_%d.pckl" % (data_folder, rank), "wb")
                )
                python_command = (
                    "python -m shotqc.run_subcircuits --data_folder %s --subcircuit_idx %d --rank %d"
                    % (data_folder, index, rank)
                )
                proc = subprocess.Popen(python_command.split(" "))
                procs.append(proc)
        [proc.wait() for proc in procs]

def modify_subcircuit(subcircuit, prep, meas):
    """
    Modify subcircuits by adding the required prepare gates and measure gates
    prep, meas: (target_qubit_index, basis)
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for index, basis in enumerate(prep):
        qubit = subcircuit.qubits[index]
        if basis == None:
            continue
        elif basis == 0: # prepare +
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 1: # prepare -
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        elif basis == 2: # prepare +i
            subcircuit_instance_dag.apply_operation_front(op=SGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 3: # prepare -i
            subcircuit_instance_dag.apply_operation_front(op=SGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        elif basis == 4: # prepare 0
            continue
        elif basis == 5: # prepare 1
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        else:
            raise Exception(f"Inital state basis ({basis}) out of range")
    for index, basis in enumerate(meas):
        qubit = subcircuit.qubits[index]
        if basis == None:
            continue
        elif basis == 0: # measure X
            subcircuit_instance_dag.apply_operation_back(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 1: # measure Y
            subcircuit_instance_dag.apply_operation_back(op=SdgGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_back(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 2:
            continue
        else:
            raise Exception("Measurement basis out of range")
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit