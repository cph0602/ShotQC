from qiskit import QuantumCircuit
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error
from helper_functions.ckt_cut import cut_circuit


### Original Circuit
original_circuit = QuantumCircuit(4)
original_circuit.h(0)
original_circuit.cx(1,2)
original_circuit.cx(0,1)
original_circuit.cx(1,2)
original_circuit.h(1)
original_circuit.x(1)
original_circuit.cx(2,3)

### Ground Truth
ground_truth = ground_truth(original_circuit)

### Cutting

subcircuits = cut_circuit(original_circuit)

### ShotQC
shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
# shotqc.print_info()
shotqc.execute(num_shots_prior=1000, num_shots_total=100000, prep_states=[0,2,4,5], use_params=True, num_iter=5)
shotqc.reconstruct()
print(shotqc.output_prob)
print("Variance: ", shotqc.variance())
print("Squared_error: ", squared_error(shotqc.output_prob, ground_truth))
