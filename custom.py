from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error
from math import pi
import numpy as np
from helper_functions.helper_instr import PseudoQPD1Q

original_circuit = QuantumCircuit(4)
original_circuit.h(0)
original_circuit.cx(1,2)
original_circuit.cx(0,1)
original_circuit.cx(1,2)
original_circuit.h(1)
original_circuit.x(1)
original_circuit.cx(2,3)
# optimization_settings = OptimizationParameters(seed=111, gate_lo=False, wire_lo=True)
# device_constraints = DeviceConstraints(qubits_per_subcircuit=4)
# cut_circuit, metadata = find_cuts(original_circuit, optimization_settings, device_constraints)
# print(cut_circuit)
ground_truth = ground_truth(original_circuit)



subcircuit_1 = QuantumCircuit(3)
subcircuit_1.h(0)
subcircuit_1.cx(1,2)
subcircuit_1.cx(0,1)
subcircuit_1.append(PseudoQPD1Q("cut_1"), [1])
subcircuit_1.append(PseudoQPD1Q("cut_2"), [2])
print(subcircuit_1)

subcircuit_2 = QuantumCircuit(2)
subcircuit_2.append(PseudoQPD1Q('cut_1'), [0])
subcircuit_2.append(PseudoQPD1Q('cut_2'), [1])
subcircuit_2.cx(0,1)
subcircuit_2.h(0)
subcircuit_2.x(0)
subcircuit_2.append(PseudoQPD1Q('cut_3'), [1])
print(subcircuit_2)

subcircuit_3 = QuantumCircuit(2)
subcircuit_3.append(PseudoQPD1Q('cut_3'), [0])
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

# sub1 = QuantumCircuit(2)
# sub2 = QuantumCircuit(2)
# # Input Subcircuit 1 #
# sub1.h(0)
# sub1.cx(0,1)
# sub1.rx(pi/4, 0)
# sub1.append(ParamCut("cut"), [1])
# # Input Subcircuit 2 #
# sub2.append(ParamCut("cut"), [0])
# sub2.x(0)
# sub2.rx(-pi/4, 1)
# sub2.cx(0,1)

# subcircuits = [sub1, sub2]

shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
# shotqc.print_info()
shotqc.execute(num_shots_prior=1000, num_shots_total=1000000, prep_states=[0,2,4,5], use_params=False, num_iter=1)
shotqc.reconstruct()
print(shotqc.output_prob)
print("Variance: ", shotqc.variance())
print("Squared_error: ", squared_error(shotqc.output_prob, ground_truth))
