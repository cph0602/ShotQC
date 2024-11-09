from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error, vector_ground_truth
from helper_functions.ckt_cut import cut_circuit
from helper_functions.helper_instr import PseudoQPD1Q
from testbench.qaoa import qaoa_circuit
from math import pi
import numpy as np
import networkx as nx
from itertools import product, combinations
from testbench.erdos import erdos20, erdos20_0, erdos20_1
import torch

org_ckt = erdos20()
sub0 = erdos20_0()
sub0.append(PseudoQPD1Q("cut_0"), [6])
sub0.append(PseudoQPD1Q("cut_1"), [7])
sub1_prepend = QuantumCircuit(14)
sub1_prepend.append(PseudoQPD1Q("cut_0"), [12])
sub1_prepend.append(PseudoQPD1Q("cut_1"), [13])
sub1 = sub1_prepend.compose(erdos20_1())
mapping = [6,7,0,1,2,3,4,8,9,10,11,12,13,14,5,15,16,17,18,19]
subcircuits = [sub0, sub1]
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
shotqc.execute(
    num_shots_prior=100, 
    num_shots_total=32000, 
    prep_states=[0,2,4,5], 
    use_params=False, 
    num_iter=5, 
    batch_size=2**19,
    distribe_shots=True
)
shotqc.reconstruct(batch_size=2**19)
print("Variance: ", shotqc.variance(batch_size=2**19))
truth = vector_ground_truth(org_ckt, mapping)
result = torch.load('output_tensor.pt', weights_only=True)
squ_error = torch.sum(torch.square(truth-result))
print(squ_error)