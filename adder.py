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
import importlib.util

# org_ckt = adder20()
# sub0_prepend = QuantumCircuit(12)
# sub0_prepend.append(PseudoQPD1Q("cut_0"), [11])
# sub0 = sub0_prepend.compose(adder20_0())
# sub0.append(PseudoQPD1Q("cut_1"), [10])
# sub1_prepend = QuantumCircuit(10)
# sub1_prepend.append(PseudoQPD1Q("cut_1"), [0])
# sub1 = sub1_prepend.compose(adder20_1())
# sub1.append(PseudoQPD1Q("cut_0"), [0])
# mapping = [i for i in range(20)]
# subcircuits = [sub0, sub1]
circuit_type = "adder"
num_qubits = 22
num_subcircuits = 2

spec = importlib.util.spec_from_file_location("config_file", f'benchmarks/{circuit_type}/config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config_dict = module.config
mapping = config_dict['mapping'][num_qubits]

org_ckt = QuantumCircuit.from_qasm_file(f'benchmarks/{circuit_type}/{circuit_type}_{num_qubits}.qasm')
subcircuits_body = []
subcircuits_prepend = []
subcircuits_append = []
subcircuits = []
for idx in range(num_subcircuits):
    subcircuits_body.append(QuantumCircuit.from_qasm_file(f'benchmarks/{circuit_type}/{circuit_type}_{num_qubits}_subcircuit_{idx}.qasm'))
    subcircuits_prepend.append(QuantumCircuit(subcircuits_body[idx].num_qubits))
    subcircuits_append.append(QuantumCircuit(subcircuits_body[idx].num_qubits))
for cut_idx, cut in enumerate(config_dict['cuts'][num_qubits]):
    subcircuits_append[cut[0][0]].append(PseudoQPD1Q(f"cut_{cut_idx}"), [cut[0][1]])
    subcircuits_prepend[cut[1][0]].append(PseudoQPD1Q(f"cut_{cut_idx}"), [cut[1][1]])
for idx in range(num_subcircuits):
    subcircuits.append(subcircuits_prepend[idx].compose(subcircuits_body[idx]).compose(subcircuits_append[idx]))


shotqc = ShotQC(
    subcircuits=subcircuits, 
    name="mycircuit", 
    verbose=True
)
# shotqc.print_info()
shotqc.execute(
    prior_ratio=0.2,
    prep_states=range(6),
    use_params=True,
    num_iter=1,
    batch_size=2**22,
    distribe_shots=True,
    ext_ratio=1
)
shotqc.reconstruct(batch_size=2**14, final_optimize=False)
print("Variance: ", shotqc.variance(batch_size=2**14))
truth = vector_ground_truth(org_ckt, mapping)
result = torch.load('shotqc/tmp_data/output_tensor.pt', weights_only=True)
squ_error = torch.sum(torch.square(truth-result))
print(squ_error)
