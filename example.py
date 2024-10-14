from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error
from helper_functions.ckt_cut import cut_circuit
from testbench.qaoa import qaoa_circuit
from math import pi
import numpy as np
import networkx as nx
from itertools import product, combinations
from testbench.supremacy import supremacy_49, supremacy_35

qc = supremacy_35()
subcircuits = cut_circuit(qc)
for idx, subcircuit in enumerate(subcircuits):
    print(f"Subcircuit {idx}: {subcircuit.num_qubits} qubits")
    print(subcircuit)