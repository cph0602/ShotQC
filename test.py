from instructions.param_cut import ParamCut, Prep_Basis, Meas_Basis
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
# from shotqc.main import ShotQC

subcircuit_1 = QuantumCircuit(3)
subcircuit_1.cx(0,1)
gate = ParamCut("cut_1")
subcircuit_1.append(gate, [1])
subcircuit_1.append(ParamCut("cut_2"), [2])
subcircuit_1.cx(0,2)
subcircuit_1.measure_all()
print(subcircuit_1)
for instr, qreg, creg in subcircuit_1.data:
    print(instr.label)

subcircuit_2 = QuantumCircuit(2)
subcircuit_2.append(ParamCut('cut_1'), [0])
subcircuit_2.cx(0,1)
subcircuit_2.append(ParamCut('cut_2'), [0])
subcircuit_2.measure_all()
print(subcircuit_2)
