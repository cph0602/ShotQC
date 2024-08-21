from instructions.param_cut import ParamCut, Prep_Basis, Meas_Basis
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from shotqc.helper import circuit_stripping

subcircuit_1 = QuantumCircuit(3)
subcircuit_1.h(0)
subcircuit_1.cx(1,2)
subcircuit_1.cx(0,1)
subcircuit_1.append(ParamCut("cut_1"), [1])
subcircuit_1.append(ParamCut("cut_2"), [2])
print(subcircuit_1)

subcircuit_2 = QuantumCircuit(2)
subcircuit_2.append(ParamCut('cut_1'), [0])
subcircuit_2.append(ParamCut('cut_2'), [1])
subcircuit_2.cx(0,1)
subcircuit_2.h(0)
subcircuit_2.x(0)
subcircuit_2.append(ParamCut('cut_3'), [1])
print(subcircuit_2)

subcircuit_3 = QuantumCircuit(2)
subcircuit_3.append(ParamCut('cut_3'), [0])
subcircuit_3.cx(0,1)
print(subcircuit_3)

# subcircuit_1 = QuantumCircuit(2)
# subcircuit_1.h(0)
# subcircuit_1.cx(0,1)
# subcircuit_1.append(ParamCut("cut_1"), [1])

# subcircuit_2 = QuantumCircuit(2)
# subcircuit_2.append(ParamCut("cut_1"), [0])
# subcircuit_2.cx(0,1)
# subcircuits = [subcircuit_1, subcircuit_2]
# subcircuit_2.append(ParamCut("cut_2"), [1])

# subcircuit_3 = QuantumCircuit(2)
# subcircuit_3.append(ParamCut("cut_2"), [0])
# subcircuit_3.cx(0,1)
print("=============================================")


subcircuits = [subcircuit_1, subcircuit_2, subcircuit_3]
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
# shotqc.print_info()
shotqc.execute(num_shots_prior=10000, num_shots_total=1000000000, prep_states = [0,2,4,5])
shotqc.reconstruct()
print(shotqc.output_prob)
