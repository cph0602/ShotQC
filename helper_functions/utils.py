from qiskit import QuantumCircuit
import re

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

def find_distinct_q_values(input_list):
    # Set to store distinct values
    distinct_values = set()
    
    # Regular expression to match 'q[number]'
    pattern = r'q\[(\d+)\]'
    
    # Iterate over the input list
    for item in input_list:
        # Find all matches in the string
        matches = re.findall(pattern, item)
        # Add the matches to the set (automatically handles distinct values)
        distinct_values.update(map(int, matches))
    
    # Return the sorted distinct values
    return sorted(distinct_values)