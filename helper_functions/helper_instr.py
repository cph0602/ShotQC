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
