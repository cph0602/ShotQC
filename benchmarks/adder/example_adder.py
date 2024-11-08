import os, math
import os, logging
from qiskit import QuantumCircuit

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Comment this line if using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from cutqc_runtime.main import CutQC # Use this just to benchmark the runtime

from cutqc.main import CutQC # Use this for exact computation

from helper_functions.benchmarks import generate_circ


if __name__ == "__main__":
    test_list = [i for i in range(20, 60)]
    for idx in test_list:
        try:
            circuit_type = "adder"
            circuit_size = idx
            circuit = generate_circ(
                num_qubits=circuit_size,
                depth=1,
                circuit_type=circuit_type,
                reg_name="q",
                connected_only=True,
                seed=None,
            )
            circuit.qasm(filename = "adder_"+str(idx)+".qasm")
            print(circuit)
            '''
            circuit = QuantumCircuit.from_qasm_file("supremacy_8.qasm")
            circuit_type = "Supremacy"
            circuit_size = circuit.num_qubits
            print("Original circuit: ")
            '''
            cutqc = CutQC(
                name="%s_%d" % (circuit_type, circuit_size),
                circuit=circuit,
                cutter_constraints={
                    "max_subcircuit_width": math.ceil(circuit.num_qubits / 4 * 3),
                    "max_subcircuit_cuts": 15,
                    "subcircuit_size_imbalance": 10,
                    "max_cuts": 15,
                    "num_subcircuits": [2, 3, 4, 5],
                },
                verbose=True,
            )
            cutqc.cut()
            if not cutqc.has_solution:
                raise Exception("The input circuit and constraints have no viable cuts")
                
            for num_subcirc in range(len(cutqc.subcircuits)):
                cutqc.subcircuits[num_subcirc].qasm(filename = "adder_"+str(idx)+"_subcircuit_"+str(num_subcirc)+".qasm")
            
            
        except:
            print(str(idx)+" can't work")
        
    
    '''
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    cutqc.build(mem_limit=32, recursion_depth=1)
    print("Cut: %d recursions." % (cutqc.num_recursions))
    print(cutqc.approximation_bins)
    '''
    '''
    
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    #cutqc.evaluate(eval_mode="qasm", num_shots_fn=test_num_shots_fn)
    cutqc.build(mem_limit=32, recursion_depth=1)
    print("Cut: %d recursions." % (cutqc.num_recursions))
    
    for (key, value) in cutqc.approximation_bins[0].items():
        print(key)
        print(value)
        print()
    
    print("The length of bins: "+str(len(cutqc.approximation_bins[0]['bins'])))
    for item in cutqc.approximation_bins[0]['bins']:
        print(item)
        
    cutqc.clean_data()
    '''
