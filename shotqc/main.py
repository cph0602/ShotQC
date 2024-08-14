from qiskit import QuantumCircuit
import os, subprocess
from typing import List
import itertools
from shotqc.executor import run_samples
from shotqc.helper import initialize_counts

class ShotQC():
    """
    Main class for shotqc
    """
    def __init__(self, subcircuits: List[QuantumCircuit] | None = None, name: str | None = None, verbose: bool = False, use_cut_params: bool = True):
        assert subcircuits != None, "Subcircuits cannot be empty"
        self.name = name
        self.subcircuits = subcircuits
        self.info = {}
        self.extract_info()
        self.verbose = verbose
        self.use_cut_params = True
        self.tmp_data_folder = "shotqc/tmp_data"
        if os.path.exists(self.tmp_data_folder):
            subprocess.run(["rm", "-r", self.tmp_data_folder])
        os.makedirs(self.tmp_data_folder)


    def extract_info(self):
        self.info["cuts"] = {}
        """Cuts: dict[cut_label] = [index for measure, index for prepare]"""
        self.subcircuits_info = []
        num_qubits = 0
        for index, subcircuit in enumerate(self.subcircuits):
            qubit_evolved = [0 for _ in range(subcircuit.num_qubits)]
            """0: not evolved; 1: evolved; -1: ended with a cut"""
            subcircuit_info = {}
            subcircuit_output = [None for _ in range(subcircuit.num_qubits)]
            subcircuit_input = [None for _ in range(subcircuit.num_qubits)]
            subcircuit_counter = {"rho": 0, "O": 0}
            subcircuit_p_cuts = {}
            subcircuit_m_cuts = {}
            subcircuit_cuts = set()
            for instr, qregs, cregs in subcircuit.data:
                assert instr.name != "measure", "No measurements are allowed in subcircuits"
                if instr.name == "param_cut":
                    cut_label = instr.label
                    assert cut_label not in subcircuit_cuts, "Same cut cannot appear in one subcircuit"
                    subcircuit_cuts.add(cut_label)
                    assert qubit_evolved[qregs[0]._index] != -1, "No gate/cut allowed after measurement cut"
                    if cut_label in self.info["cuts"].keys():
                        if qubit_evolved[qregs[0]._index] == 0:
                            assert self.info["cuts"][cut_label][1] == None, f"Repeat preparation for cut {cut_label}"
                            qubit_evolved[qregs[0]._index] = 1
                            subcircuit_input[qregs[0]._index] = cut_label
                            instr.params[0] = 'p'
                            subcircuit_counter["rho"] += 1
                            self.info["cuts"][cut_label][1] = index
                        elif qubit_evolved[qregs[0]._index] == 1:
                            assert self.info["cuts"][cut_label][0] == None, f"Repeat measurement for cut {cut_label}"
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
                            instr.params[0] = 'm'
                            subcircuit_counter["O"] += 1
                            self.info["cuts"][cut_label][0] = index
                    else:
                        if qubit_evolved[qregs[0]._index] == 0:
                            qubit_evolved[qregs[0]._index] = 1
                            subcircuit_input[qregs[0]._index] = cut_label
                            instr.params[0] = 'p'
                            subcircuit_counter["rho"] += 1
                            self.info["cuts"][cut_label] = [None, index]
                        elif qubit_evolved[qregs[0]._index] == 1:
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
                            instr.params[0] = 'm'
                            subcircuit_counter["O"] += 1
                            self.info["cuts"][cut_label] = [index, None]
                else:
                    for qreg in qregs:
                        assert qubit_evolved[qreg._index] != -1, "No gate/cut allowed after measurement cut"
                        qubit_evolved[qreg._index] = 1
            subcircuit_counter['effective'] = subcircuit.num_qubits - subcircuit_counter["O"]
            for i in range(subcircuit.num_qubits):
                if subcircuit_input[i] != None:
                    subcircuit_p_cuts[i] = subcircuit_input[i]
                if subcircuit_output[i] != None:
                    subcircuit_m_cuts[i] = subcircuit_output[i]
            subcircuit_info['input'] = subcircuit_input
            subcircuit_info['output'] = subcircuit_output
            subcircuit_info['counter'] = subcircuit_counter
            subcircuit_info['p_cuts'] = subcircuit_p_cuts
            subcircuit_info['m_cuts'] = subcircuit_m_cuts
            self.subcircuits_info.append(subcircuit_info)
            num_qubits += subcircuit_counter["effective"]
        for key in self.info["cuts"].keys():
            assert self.info["cuts"][key][0] != None and self.info["cuts"][key][1] != None, f"Cut {key} does not form a pair"
        self.info["num_subcircuits"] = len(self.subcircuits)
        self.info["num_qubits"] = num_qubits
        self.info["num_cuts"] = len(self.info["cuts"])


    def print_info(self):
        for circuit in self.subcircuits:
            print(circuit)
        print(self.subcircuits_info)
        print(self.info)


    def clean_data(self):
        subprocess.run(["rm", "-r", self.tmp_data_folder])
    
    
    def execute(self, num_shots_prior: int | None = None, num_shots_total: int | None = None, num_iter: int = 1, prep_states: List[int] = range(6), run_mode: str = "qasm"):
        """
        Run circuits with optimal shot distribution and optimal cut parameters.
        run_mode = "qasm": classical simulator
        num_shots_prior: number of prior shots for each subcircuit
        num_shots_total: total number of shots
        num_iter: total iteration rounds
        """
        self.prep_states = prep_states
        print(f"--> Generating subcircuit entries:")
        self._generate_subcircuit_entries()
        for i in range(self.info["num_subcircuits"]):
            print(f"Subcircuit {i} has {self.info["num_subcircuits_entries"][i]} entries.")
        assert num_shots_prior*self.info["num_total_entries"] <= num_shots_total, f"Total number of prior shots({num_shots_prior*self.info["num_subcircuits_entries"]}) cannot exceed total shots({num_shots_total})."
        initialize_counts(self.tmp_data_folder, self.subcircuits, self.subcircuits_entries)
        self._run_prior_samples(num_shots_prior)


    def _run_prior_samples(self, num_shots_prior: int):
        if self.verbose:
            print("--> Running Prior Samples")
        run_samples(self.subcircuits, self.subcircuits_entries, "qasm", "equal", num_shots_prior, self.tmp_data_folder)


    def _generate_subcircuit_entries(self):
        self.subcircuits_entries = []
        num_subcircuits_entries = []
        num_total_entries = 0
        for index, subcircuit in enumerate(self.subcircuits):
            subcircuit_entries = []
            prep = [None for _ in range(subcircuit.num_qubits)]
            meas = [None for _ in range(subcircuit.num_qubits)]
            p_qubits = list(self.subcircuits_info[index]['p_cuts'].keys())
            m_qubits = list(self.subcircuits_info[index]['m_cuts'].keys())
            for circuit_prepare in itertools.product(self.prep_states, repeat=self.subcircuits_info[index]['counter']['rho']):
                for circuit_measure in itertools.product(range(3), repeat=self.subcircuits_info[index]['counter']['O']):
                    # print(circuit_prepare[0], circuit_measure[0])
                    for i in range(self.subcircuits_info[index]['counter']['rho']):
                        prep[p_qubits[i]] = circuit_prepare[i]
                    for i in range(self.subcircuits_info[index]['counter']['O']):
                        meas[m_qubits[i]] = circuit_measure[i]
                    subcircuit_entries.append((prep[:], meas[:]))
                    # subcircuit_variant = modify_subcircuit(subcircuit, prep, meas)
            self.subcircuits_entries.append(subcircuit_entries)
            num_subcircuits_entries.append(len(subcircuit_entries))
            num_total_entries += len(subcircuit_entries)
        self.info["num_subcircuits_entries"] = num_subcircuits_entries
        self.info["num_total_entries"] = num_total_entries
        print(self.subcircuits_entries)
        