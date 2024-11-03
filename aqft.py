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
from testbench.aqft import aqft20, aqft20_0, aqft20_1
import torch

org_ckt = aqft20()
sub0_prepend = QuantumCircuit(15)
sub0 = sub0_prepend.compose(aqft20_0())
sub0.append(PseudoQPD1Q("cut_0"), [4])
sub0.append(PseudoQPD1Q("cut_1"), [3])
sub0.append(PseudoQPD1Q("cut_2"), [2])
sub0.append(PseudoQPD1Q("cut_3"), [1])
sub0.append(PseudoQPD1Q("cut_4"), [0])
sub1_prepend = QuantumCircuit(10)
sub1_prepend.append(PseudoQPD1Q("cut_0"), [9])
sub1_prepend.append(PseudoQPD1Q("cut_1"), [8])
sub1_prepend.append(PseudoQPD1Q("cut_2"), [7])
sub1_prepend.append(PseudoQPD1Q("cut_3"), [6])
sub1_prepend.append(PseudoQPD1Q("cut_4"), [5])
sub1 = sub1_prepend.compose(aqft20_1())
mapping = [10,11,12,13,14,15,16,17,18,19,0,1,2,3,4,5,6,7,8,9]
subcircuits = [sub0, sub1]
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
shotqc.print_info()
shotqc.execute(
    num_shots_prior=100, 
    num_shots_total=2000000, 
    prep_states=[0,2,4,5], 
    use_params=False, 
    num_iter=1, 
    batch_size=2**19,
    distribe_shots=False
)
shotqc.reconstruct(batch_size=2**19)
print("Variance: ", shotqc.variance(batch_size=2**19))
truth = vector_ground_truth(org_ckt, mapping)
result = torch.load('output_tensor.pt', weights_only=True)
squ_error = torch.sum(torch.square(truth-result))
print(squ_error)