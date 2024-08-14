from instructions.param_cut import ParamCut, Prep_Basis, Meas_Basis
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from shotqc.helper import circuit_stripping

subcircuit_1 = QuantumCircuit(3)
subcircuit_1.cx(0,1)
gate = ParamCut("cut_1")
subcircuit_1.append(gate, [1])
subcircuit_1.append(ParamCut("cut_2"), [2])
subcircuit_1.cx(0,2)
print(subcircuit_1)

subcircuit_2 = QuantumCircuit(2)
subcircuit_2.append(ParamCut('cut_1'), [0])
subcircuit_2.cx(0,1)
subcircuit_2.append(ParamCut('cut_2'), [0])
print(subcircuit_2)

print("=============================================")


subcircuits = [subcircuit_1, subcircuit_2]
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
shotqc.print_info()
shotqc.execute(100, 9000)