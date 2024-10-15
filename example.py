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
from testbench.supremacy import supremacy_25
from qiskit_addon_cutting.

qc = supremacy_25()
subcircuit_1 = {
    "prep": [5,10,12,14],
    "meas": [11]
}
subcircuit_2 = {
    "prep": [3],
    "meas": [0,2,4,9]
}
mapping = [0,1,2,3,4,5,6,7,8,9,14,10,15,11,12,16,17,18,19,13,20,21,22,23]
