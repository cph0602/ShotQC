from instructions.param_cut import ParamCut, Prep_Basis, Meas_Basis
from qiskit import QuantumCircuit

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.append(ParamCut("Cut_1"), [1])
