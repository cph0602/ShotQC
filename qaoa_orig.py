from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import qiskit_aer as aer
# Define the MaxCut problem instance (graph)
graph = {
    'edges': [(0, 1), (1, 2), (2, 3), (3, 0)],
    'weights': [1, 2, 3, 1]
}

# Create the QAOA circuit
def qaoa_circuit(gamma, beta, graph):
    n = len(graph['weights'])
    qc = QuantumCircuit(n, n)
    
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

# Evaluate the QAOA circuit
def evaluate_qaoa(gamma, beta, graph, n_shots=1000):
    qc = qaoa_circuit(gamma, beta, graph)
    qc.measure(range(len(graph['weights'])), range(len(graph['weights'])))
    
    aer_sim = aer.Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, aer_sim)
    qobj = assemble(transpiled_qc)
    
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(qc)
    
    # Calculate the objective function value
    obj_val = 0
    for bit_string, count in counts.items():
        bit_weight = sum([int(bit) * weight for bit, weight in zip(bit_string, graph['weights'])])
        obj_val += bit_weight * count / n_shots
    
    return obj_val

# Optimize QAOA parameters
def optimize_qaoa(graph, n_shots=1000, n_steps=50):
    # Initial parameters
    gamma = np.random.uniform(0, np.pi, n_steps)
    beta = np.random.uniform(0, np.pi, n_steps)
    
    # Optimization loop
    for step in range(n_steps):
        obj_val = evaluate_qaoa(gamma[step], beta[step], graph, n_shots=n_shots)
        print(f"Step {step + 1}/{n_steps} - Objective Value: {obj_val}")
    
    return gamma, beta

# Run the optimization
gamma_opt, beta_opt = optimize_qaoa(graph)
print(f"Optimal gamma: {gamma_opt[-1]}")
print(f"Optimal beta: {beta_opt[-1]}")