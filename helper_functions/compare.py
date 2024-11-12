import random
from qiskit import QuantumCircuit
import qiskit_aer as aer
from qiskit.quantum_info import Statevector
import copy, itertools, torch

def ground_truth(original_circuit, mapping = None):
    if mapping == None:
        mapping = [i for i in range(original_circuit.num_qubits)]
    circuit = copy.deepcopy(original_circuit)
    simulator = aer.Aer.get_backend("statevector_simulator")
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)
    prob_vector = Statevector(statevector).probabilities()
    output_dict = {}
    for i in range(len(prob_vector)):
        bits = "0"*(circuit.num_qubits - (len(bin(i))-2)) + str(bin(i)[2:])
        bits = "".join([bits[circuit.num_qubits - 1 - mapping[circuit.num_qubits - 1 - i]] for i in range(circuit.num_qubits)])
        output_dict[bits] = prob_vector[i]
    return output_dict

def squared_error(output_prob, ground_truth):
    error = 0
    for key in output_prob:
        error += (output_prob[key] - ground_truth[key])**2
    return error

def vector_ground_truth(original_circuit, mapping = None, device=None):
    if mapping == None:
        mapping = [i for i in range(original_circuit.num_qubits)]
    qiskit_permute = [i for i in range(original_circuit.num_qubits)][::-1]
    circuit = copy.deepcopy(original_circuit)
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = aer.Aer.get_backend("statevector_simulator")
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)
    prob_vector = Statevector(statevector).probabilities()
    truth = torch.tensor(prob_vector).view((2,)*original_circuit.num_qubits).permute(qiskit_permute).permute(tuple(mapping)).permute(qiskit_permute).flatten().to(device)
    return truth