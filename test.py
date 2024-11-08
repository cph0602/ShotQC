# from helper_functions.utils import find_distinct_q_values
# from qiskit import QuantumCircuit

# from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
# from qiskit.compiler import transpile
 
# service = QiskitRuntimeService(
#     channel='ibm_quantum',
#     instance='ibm-q-hub-ntu/jiang-jie-hong/default',
#     token='88213fbca00c9fe8ac6102df566e4e7bcafeb0854a3a44a4bf063d821c49536aad79823896bec4cd6f6eaca3261f49579dcd3f747a4130b2d11bfa109f80d8f2'
# )
# backend = service.least_busy(operational=True, simulator=False)


# # Create a quantum circuit
# qc = QuantumCircuit(2)
# qc.h(0)  # Apply Hadamard gate to qubit 0
# qc.cx(0, 1)  # Apply CNOT gate with control=0, target=1
# qc.measure_all
# sampler = Sampler(backend)
# print(qc.cregs)
# job = sampler.run(transpile([qc], backend))
# print(f"job id: {job.job_id()}")
# result = job.result()
# print(result)
import torch
a = torch.load("shotqc/tmp_data/subcircuit_0_entry_0.pt")
print(a[a>1e-3])