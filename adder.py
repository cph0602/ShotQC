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
from testbench.adder import adder20, adder20_0, adder20_1
import torch

org_ckt = adder20()
sub0_prepend = QuantumCircuit(12)
sub0_prepend.append(PseudoQPD1Q("cut_0"), [11])
sub0 = sub0_prepend.compose(adder20_0())
sub0.append(PseudoQPD1Q("cut_1"), [10])
sub1_prepend = QuantumCircuit(10)
sub1_prepend.append(PseudoQPD1Q("cut_1"), [0])
sub1 = sub1_prepend.compose(adder20_1())
sub1.append(PseudoQPD1Q("cut_0"), [0])
mapping = [i for i in range(20)]
subcircuits = [sub0, sub1]
shotqc = ShotQC(
    subcircuits=subcircuits, 
    name="mycircuit", 
    verbose=True
)
shotqc.execute(
    num_shots_prior=100, 
    num_shots_total=32000, 
    prep_states=[0,2,4,5], 
    use_params=True, 
    num_iter=1,
    batch_size=2**20,
    distribe_shots=True
)
shotqc.reconstruct(batch_size=2**20)
print("Variance: ", shotqc.variance(batch_size=2**20))
truth = vector_ground_truth(org_ckt, mapping)
result = torch.load('output_tensor.pt', weights_only=True)
squ_error = torch.sum(torch.square(truth-result))
print(squ_error)