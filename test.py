# from qiskit import QuantumCircuit
# from qiskit.circuit import Instruction
# from shotqc.main import ShotQC
# from helper_functions.compare import ground_truth, squared_error
# from math import pi
# import numpy as np
# from qiskit_addon_cutting.qpd.instructions.qpd_gate import SingleQubitQPDGate

# original_circuit = QuantumCircuit(4)
# original_circuit.h(0)
# original_circuit.cx(1,2)
# original_circuit.cx(0,1)
# original_circuit.cx(1,2)
# original_circuit.h(1)
# original_circuit.x(1)
# original_circuit.cx(2,3)
# # optimization_settings = OptimizationParameters(seed=111, gate_lo=False, wire_lo=True)
# # device_constraints = DeviceConstraints(qubits_per_subcircuit=4)
# # cut_circuit, metadata = find_cuts(original_circuit, optimization_settings, device_constraints)
# # print(cut_circuit)
# ground_truth = ground_truth(original_circuit)



# subcircuit_1 = QuantumCircuit(3)
# subcircuit_1.h(0)
# subcircuit_1.cx(1,2)
# subcircuit_1.cx(0,1)
# subcircuit_1.append(SingleQubitQPDGate("cut_1", 0), [1])
# subcircuit_1.append(SingleQubitQPDGate("cut_2", 0), [2])
# print(subcircuit_1)

# subcircuit_2 = QuantumCircuit(2)
# subcircuit_2.append(SingleQubitQPDGate('cut_1', 0), [0])
# subcircuit_2.append(SingleQubitQPDGate('cut_2', 0), [1])
# subcircuit_2.cx(0,1)
# subcircuit_2.h(0)
# subcircuit_2.x(0)
# subcircuit_2.append(SingleQubitQPDGate('cut_3', 0), [1])
# print(subcircuit_2)

# subcircuit_3 = QuantumCircuit(2)
# subcircuit_3.append(SingleQubitQPDGate('cut_3', 0), [0])
# subcircuit_3.cx(0,1)
# print(subcircuit_3)

# # subcircuit_1 = QuantumCircuit(2)
# # subcircuit_1.h(0)
# # subcircuit_1.cx(0,1)
# # subcircuit_1.append(ParamCut("cut_1"), [1])

# # subcircuit_2 = QuantumCircuit(2)
# # subcircuit_2.append(ParamCut("cut_1"), [0])
# # subcircuit_2.cx(0,1)
# # subcircuits = [subcircuit_1, subcircuit_2]
# # subcircuit_2.append(ParamCut("cut_2"), [1])

# # subcircuit_3 = QuantumCircuit(2)
# # subcircuit_3.append(ParamCut("cut_2"), [0])
# # subcircuit_3.cx(0,1)
# print("=============================================")


# subcircuits = [subcircuit_1, subcircuit_2, subcircuit_3]

# # sub1 = QuantumCircuit(2)
# # sub2 = QuantumCircuit(2)
# # # Input Subcircuit 1 #
# # sub1.h(0)
# # sub1.cx(0,1)
# # sub1.rx(pi/4, 0)
# # sub1.append(ParamCut("cut"), [1])
# # # Input Subcircuit 2 #
# # sub2.append(ParamCut("cut"), [0])
# # sub2.x(0)
# # sub2.rx(-pi/4, 1)
# # sub2.cx(0,1)

# # subcircuits = [sub1, sub2]

# shotqc = ShotQC(subcircuits=subcircuits, name="mycircuit", verbose=True)
# # shotqc.print_info()
# shotqc.execute(num_shots_prior=1000, num_shots_total=1000000, prep_states=[0,2,4,5], use_params=False, num_iter=5)
# shotqc.reconstruct()
# print(shotqc.output_prob)
# print("Variance: ", shotqc.variance())
# print("Squared_error: ", squared_error(shotqc.output_prob, ground_truth))

# import torch

# def multiply_with_one_off(tensor, bitstring):
#     # Step 1: Convert bitstring to tensor index
#     index = torch.tensor([int(b) for b in bitstring], dtype=torch.long).to('cuda')

#     # Step 2: Create a mask that flips each bit
#     n = len(bitstring)
#     flip_mask = torch.eye(n, dtype=torch.long).to('cuda')  # Identity matrix for flipping each bit

#     # Step 3: Compute the flipped indices by XORing with the mask
#     flipped_indices = index.unsqueeze(0).repeat(n, 1) ^ flip_mask  # XOR flips the bits

#     # Step 4: Gather the values from the tensor for both the original and flipped indices
#     original_value = tensor[tuple(index.tolist())]  # Value at the original index
#     flipped_values = tensor[tuple(flipped_indices.T.tolist())]  # Values at the flipped indices

#     # Step 5: Multiply the original value with the values at flipped indices
#     result = original_value * flipped_values

#     return result

# x = torch.zeros((3,3,2,2))
# x[2,1,0,1] = 1 # ZY01
# y = x.permute((0,2, 1,3)).contiguous()
# print(y.shape)
# z = y.view((6,6))
# print(z[4,3]) # 0, -i

# num_meas=2
# permute_tuple = tuple([i//2 if i%2 == 0 else i//2+num_meas for i in range(2*num_meas)])
# print(permute_tuple)
import torch

def insert_elements(original_tensor, new_elements, insert_indices):
    # Get the total size of the new tensor
    total_size = original_tensor.size(0) + new_elements.size(0)

    # Create an empty tensor to hold the result
    result_tensor = torch.empty(total_size, dtype=original_tensor.dtype, device=original_tensor.device)

    # Create a mask to determine where to insert new elements
    mask = torch.zeros(total_size, dtype=torch.bool, device=original_tensor.device)
    mask[insert_indices] = True  # Set positions for the new elements

    # Get the original indices (where the original elements will go)
    original_indices = torch.arange(total_size, device=original_tensor.device)[~mask]

    # Insert the new elements into the result tensor
    result_tensor[mask] = new_elements
    result_tensor[original_indices] = original_tensor

    return result_tensor

# Example usage
original_tensor = torch.tensor([1, 2, 3, 4, 5], device='cuda')  # Example original tensor
new_elements = torch.tensor([99, 100], device='cuda')  # New elements to insert
insert_indices = torch.tensor([1, 3], device='cuda')  # Indices where new elements will be inserted

# Insert elements
result_tensor = insert_elements(original_tensor, new_elements, insert_indices)

print(result_tensor)  # Should show the original tensor with new elements inserted

import torch
from shotqc.helper import generate_all_bitstrings
# Given batch of bitstrings
import torch

# Given batch of bitstrings
bitstrings = torch.tensor([[0, 0], [0, 1]])

# Initialize a tensor of zeros with the shape (bitstrings.shape[0], 2, 2)
output_tensor = torch.zeros(bitstrings.shape[0], 2, 2)

# Use advanced indexing to set ones at the specified positions
batch_indices = torch.arange(bitstrings.shape[0])  # Create an index for the batch dimension
output_tensor[batch_indices, bitstrings[:, 0], bitstrings[:, 1]] = 1

print(output_tensor)

bit = generate_all_bitstrings(2)
print(tuple(bit.t()))
a = torch.zeros((4,3,2,2,2))
a[0,0,0,0,0] = 1
a[1,0,0,1,0] = 2
a[2,0,1,0,0] = 3
a[3,0,1,1,0] = 4
output_tensor = torch.zeros(bit.shape[0], 2, 2)
b = torch.rand((3,2))
print(b)
# Use advanced indexing to set ones at the specified positions
batch_indices = torch.arange(bit.shape[0])  # Create an index for the batch dimension
output_tensor[(batch_indices,)+tuple(bit.t())] = 1
output_tensor = output_tensor.view(4,1,2,2,1).repeat(1,3,1,1,2)
print(output_tensor.shape)
b = b.view(1,3,1,1,2).repeat(4,1,2,2,1)
x = b * output_tensor
print(x[0,:,0,0,:])