# how to read instr (note to self)

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