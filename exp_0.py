from qiskit import QuantumCircuit
from shotqc.main import ShotQC
from helper_functions.compare import vector_ground_truth
from helper_functions.helper_instr import PseudoQPD1Q
import torch
import importlib.util

circuit_type = "regular"
num_qubits = 22
num_subcircuits = 2
name = circuit_type + "_0"
device = "cuda:0"
batch_size = 2**17

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
    name=name, 
    verbose=True,
    device=device
)
# shotqc.print_info()
shotqc.execute(
    prior_ratio=0.2,
    prep_states=[0,2,4,5],
    use_params=False,
    num_iter=5,
    batch_size=batch_size,
    distribe_shots=True,
    ext_ratio=10
)
shotqc.reconstruct(batch_size=batch_size, final_optimize=False)
print("Variance: ", shotqc.variance(batch_size=batch_size))
truth = vector_ground_truth(org_ckt, mapping, device)
result = torch.load(f'shotqc/{name}_tmp_data/output_tensor.pt', weights_only=True)
squ_error = torch.sum(torch.square(truth-result))
print(squ_error)
# print(truth)
print(torch.sum(torch.square(truth)))
