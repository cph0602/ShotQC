from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints
)
from qiskit_addon_cutting.wire_cutting_transforms import cut_wires
from qiskit_addon_cutting.cutting_decomposition import partition_problem
from helper_functions.utils import fix_circuit
from math import floor

def cut_circuit(original_circuit):
    optimization_settings = OptimizationParameters(seed=111, gate_lo=False, wire_lo=True)
    device_constraints = DeviceConstraints(qubits_per_subcircuit=floor(3 * original_circuit.num_qubits / 4))
    cut_circuit, metadata = find_cuts(original_circuit, optimization_settings, device_constraints)
    qc_1 = cut_wires(cut_circuit)
    partitioned_problem = partition_problem(circuit=qc_1)
    return fix_circuit(partitioned_problem.subcircuits)