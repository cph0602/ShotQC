from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister

class PseudoQPD1Q(Instruction):
    def __init__(self, label=None):
        # The instruction acts on one qubit, has no parameters, and is named "qpd_1q"
        super().__init__("qpd_1q", 1, 0, params=[], label=label)

    def _define(self):
        # Define what the instruction does (nothing in this case)
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        qc = QuantumCircuit(1)
        self.definition = qc

# Create the custom instruction and a circuit to use it
qpd_1q = PseudoQPD1Q(label="custom_label")
qc = QuantumCircuit(1)
qc.append(qpd_1q, [0])

print(qc)