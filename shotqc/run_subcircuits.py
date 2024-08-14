import argparse, pickle, subprocess
from shotqc.executor import modify_subcircuit
from qiskit import QuantumCircuit
import qiskit_aer as aer
import copy, psutil, os, itertools
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--data_folder", metavar="S", type=str)
    parser.add_argument("--subcircuit_idx", metavar="N", type=int)
    parser.add_argument("--rank", metavar="N", type=int)
    args = parser.parse_args()

    subcircuit_idx = args.subcircuit_idx
    meta_info = pickle.load(open("%s/meta_info.pckl" % (args.data_folder), "rb"))
    subcircuit = meta_info["subcircuits"][subcircuit_idx]
    run_mode = meta_info["run_mode"]
    SD_dist = meta_info["SD_dist"][subcircuit_idx]
    subcircuit_entries = meta_info["subcircuits_entries"][subcircuit_idx]
    rank_jobs = pickle.load(
        open("%s/rank_%d.pckl" % (args.data_folder, args.rank), "rb")
    )

    max_memory_mb = psutil.virtual_memory().total >> 20
    max_memory_mb = int(max_memory_mb / 4 * 3)
    for job in rank_jobs:
        subcircuit_entry = subcircuit_entries[job]
        circuit = modify_subcircuit(subcircuit, subcircuit_entry[0], subcircuit_entry[1])
        num_shots = SD_dist[job]
        if run_mode == "qasm":
            simulator = aer.Aer.get_backend('aer_simulator', max_memory_mb=max_memory_mb)
            circuit.measure_all()
            sim_job = simulator.run(circuit, shots = num_shots)
            sim_result = sim_job.result()
            sim_counts = sim_result.get_counts(circuit)
        else:
            raise Exception("Run mode not supported")

        previous_counts = pickle.load(open("%s/subcircuit_%d_entry_%d.pckl" % (args.data_folder, subcircuit_idx, job), "rb"))
        previous_counts["total_shots"] += num_shots
        for bitstring in sim_counts:
            previous_counts["counts"][bitstring] += sim_counts[bitstring]
        pickle.dump(
            {
                "total_shots": previous_counts["total_shots"],
                "counts": previous_counts["counts"],
            },
            open("%s/subcircuit_%d_entry_%d.pckl" % (args.data_folder, subcircuit_idx, job), "wb"),
        )

        
        


    