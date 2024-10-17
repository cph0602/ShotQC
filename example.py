from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error
from helper_functions.ckt_cut import cut_circuit
from helper_functions.helper_instr import PseudoQPD1Q
from testbench.qaoa import qaoa_circuit
from math import pi
import numpy as np
import networkx as nx
from itertools import product, combinations
from testbench.supremacy import supremacy_25, sup25_sub0, sup25_sub1

qc = supremacy_25()
subcircuit_0 = {
    "prep": [5,10,12,14],
    "meas": [11]
}
subcircuit_1 = {
    "prep": [3],
    "meas": [0,2,4,9]
}
mapping = [0,1,2,3,4,5,6,7,8,9,14,10,15,11,12,16,17,18,19,13,20,21,22,23]

sub0_prepend = QuantumCircuit(15)
sub0_prepend.append(PseudoQPD1Q("cut_0"), [5])
sub0_prepend.append(PseudoQPD1Q("cut_1"), [10])
sub0_prepend.append(PseudoQPD1Q("cut_3"), [12])
sub0_prepend.append(PseudoQPD1Q("cut_4"), [14])
sub1_prepend = QuantumCircuit(15)
sub1_prepend.append(PseudoQPD1Q("cut_2"), [3])
sub0 = sub0_prepend.compose(sup25_sub0())
sub1 = sub1_prepend.compose(sup25_sub1())
sub0.append(PseudoQPD1Q("cut_2"), [11])
sub1.append(PseudoQPD1Q("cut_0"), [0])
sub1.append(PseudoQPD1Q("cut_1"), [2])
sub1.append(PseudoQPD1Q("cut_3"), [4])
sub1.append(PseudoQPD1Q("cut_4"), [9])

subcircuits = [sub0, sub1]
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True, reset_files=False)
shotqc.execute(
    num_shots_prior=100, 
    num_shots_total=1000000, 
    prep_states=[0,2,4,5], 
    use_params=False, 
    num_iter=1, 
    batch_size=16, 
    distribe_shots=False,
    debug=True
)
# shotqc.reconstruct()
# print(shotqc.output_prob)
# print("Variance: ", shotqc.variance())
# original_ckt = supremacy_25()
# ground_truth = ground_truth(original_ckt)
# print("Squared_error: ", squared_error(shotqc.output_prob, ground_truth))