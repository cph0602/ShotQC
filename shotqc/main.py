from qiskit import QuantumCircuit
import os, subprocess
from typing import List
import itertools
from helper_functions.helper import modify_subcircuit

class ShotQC():
    """
    Main class for shotqc
    """
    def __init__(self, subcircuits: List[QuantumCircuit] | None = None, name: str | None = None, verbose: bool = False):
        assert subcircuits != None, "Subcircuits cannot be empty"
        self.name = name
        self.subcircuits = subcircuits
        self.info = {}
        self.extract_info()
        self.verbose = verbose
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
            subcircuit_p_cuts = []
            subcircuit_m_cuts = []
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
                if subcircuit_input != None:
                    subcircuit_p_cuts.append((cut_label, i))
                if subcircuit_output != None:
                    subcircuit_m_cuts.append((cut_label, i))
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
    def _run_subcircuits(self, mode: str = "Equal", distribution: List[int] | None = None, num_shots: int | None = None):
        """
        Run Subcircuits in equal mode or unequal mode
        Equal: run each subcircuit num_shots times
        Distribute: run each subcircuit according to the distribution list
        """
        
    def run(self, prep_states: List[int] = range(6)):
        self.prep_states = prep_states
        self.generate_subcircuit_variants()

    def generate_subcircuit_variants(self):
        self.subcircuits_variants = []
        self.subcircuits_entries = []
        for index, subcircuit in enumerate(self.subcircuits):
            print(index)
            subcircuit_variants = []
            subcircuit_entries = []
            for circuit_prepare in itertools.product(self.prep_states, repeat=self.subcircuits_info[index]['counter']['rho']):
                for circuit_measure in itertools.product(range(3), repeat=self.subcircuits_info[index]['counter']['O']):
                    prep = [(self.subcircuits_info[index]['p_cuts'][i][1], circuit_prepare[i]) for i in range(self.subcircuits_info[index]['counter']['rho'])]
                    meas = [(self.subcircuits_info[index]['m_cuts'][i][1], circuit_measure[i]) for i in range(self.subcircuits_info[index]['counter']['O'])]
                    subcircuit_entries.append((circuit_prepare, circuit_measure))
                    subcircuit_variant = modify_subcircuit(subcircuit, prep, meas)
                    subcircuit_variants.append(subcircuit_variant)
            self.subcircuits_variants.append(subcircuit_variants)
            self.subcircuits_entries.append(subcircuit_entries)
        for i in range(2):
            for circuit in self.subcircuits_variants[i]:
                print(circuit)