from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import copy, pickle
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

def modify_subcircuit(subcircuit, prep, meas):
    """
    Modify subcircuits by adding the required prepare gates and measure gates
    prep, meas: (target_qubit_index, basis)
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for p in prep:
        print("p: ", p)
        qubit = subcircuit.qubits[p[0]]
        basis = p[1]
        if basis == 0: # prepare +
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 1: # prepare -
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        elif basis == 2: # prepare +i
            subcircuit_instance_dag.apply_operation_front(op=SGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 3: # prepare -i
            subcircuit_instance_dag.apply_operation_front(op=SGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=HGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        elif basis == 4: # prepare 0
            continue
        elif basis == 5: # prepare 1
            subcircuit_instance_dag.apply_operation_front(op=XGate(), qargs=[qubit], cargs=[])
        else:
            raise Exception("Inital state basis out of range")
    for m in meas:
        qubit = subcircuit.qubits[m[0]]
        basis = m[1]
        if basis == 0: # measure X
            subcircuit_instance_dag.apply_operation_back(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 1: # measure Y
            subcircuit_instance_dag.apply_operation_back(op=SdgGate(), qargs=[qubit], cargs=[])
            subcircuit_instance_dag.apply_operation_back(op=HGate(), qargs=[qubit], cargs=[])
        elif basis == 2:
            continue
        else:
            raise Exception("Measurement basis out of range")
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit