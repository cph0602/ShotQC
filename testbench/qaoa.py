from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import qiskit_aer as aer

# Create the QAOA circuit
def qaoa_circuit(gamma, beta, graph):
    n = len(graph['weights'])
    qc = QuantumCircuit(n)
    
    # Apply Hadamard gate to all qubits
    qc.h(range(n))
    
    # Apply the Ising-type gates for each edge
    for edge, weight in zip(graph['edges'], graph['weights']):
        qc.cx(edge[0], edge[1])
        qc.u(2 * gamma * weight, 0, 0, edge[1])
        qc.cx(edge[0], edge[1])
    
    # Apply single qubit X-rotations
    qc.rx(2 * beta, range(n))
    
    return qc
