from qiskit import QuantumCircuit

def fix_circuit(subcircuits):
    new_subcircuits = []
    for subcircuit_idx in range(len(subcircuits)):
        subcircuit = subcircuits[subcircuit_idx]
        new_subcircuit = QuantumCircuit(subcircuit.num_qubits)
        for instr, qregs, cregs in subcircuit.data:
            corrected_qregs = [new_subcircuit.qubits[subcircuit.qubits.index(q)] for q in qregs]
            new_subcircuit.append(instr, corrected_qregs, cregs)
        new_subcircuits.append(new_subcircuit)
        # print("index: ", subcircuit_idx)
        # print(new_subcircuit)
    return new_subcircuits