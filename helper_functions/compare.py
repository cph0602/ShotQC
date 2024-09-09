import random
from qiskit import QuantumCircuit
import qiskit_aer as aer
from qiskit.quantum_info import Statevector
import copy, itertools

def ground_truth(original_circuit):
    circuit = copy.deepcopy(original_circuit)
    simulator = aer.Aer.get_backend("statevector_simulator")
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)
    prob_vector = Statevector(statevector).probabilities()
    output_dict = {}
    for i in range(len(prob_vector)):
        bits = "0"*(circuit.num_qubits - (len(bin(i))-2)) + str(bin(i)[2:])
        output_dict[bits] = prob_vector[i]
    return output_dict

def squared_error(output_prob, ground_truth):
    error = 0
    for key in output_prob:
        error += (output_prob[key] - ground_truth[key])**2
    return error