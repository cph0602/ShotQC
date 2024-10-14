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


graph = {
    'edges': [(0, 1), (2, 5), (3, 7), (4, 9), (6, 8), (1, 4), (2, 3), (5, 6), (7, 9), (0, 8)],
    'weights': [4, 6, 2, 5, 3, 8, 7, 1, 9, 2]
}
gamma = 1.324
beta = 0.78
original_circuit = qaoa_circuit(gamma, beta, graph)
# print(original_circuit)

### Ground Truth
mapping = [1, 4, 7, 9, 0, 2, 3, 5, 6, 8]
ground_truth = ground_truth(original_circuit, mapping)
# print(ground_truth)

### Cutting

# print(original_circuit)
subcircuits = cut_circuit(original_circuit)

### ShotQC
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
shotqc.print_info()
shotqc.execute(
    num_shots_prior=100, 
    num_shots_total=24000, 
    prep_states=[0,2,4,5], 
    use_params=True, 
    num_iter=1,
    run_mode="qasm"
)
shotqc.reconstruct()
# print(shotqc.output_prob)
print("Variance: ", shotqc.variance())
print("Squared_error: ", squared_error(shotqc.output_prob, ground_truth))
# print(ground_truth)