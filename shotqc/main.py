from qiskit import QuantumCircuit
import os, subprocess, random, pickle, itertools
from typing import List
from math import floor
from shotqc.executor import run_samples
from shotqc.helper import initialize_counts, read_probs_with_prior, params_matrix_to_list
from shotqc.optimizer import optimize_params
from shotqc.postprocesser import postprocess
from shotqc.overhead import generate_distribution, calculate_variance

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
        # if os.path.exists(self.tmp_data_folder):
        #     subprocess.run(["rm", "-r", self.tmp_data_folder])
        # os.makedirs(self.tmp_data_folder)


    def execute(self, num_shots_prior: int | None = 1024, num_shots_total: int | None = None, 
                num_iter: int = 1, prep_states: List[int] = range(6), run_mode: str = "qasm", 
                use_params: bool = True, method: str = "L-BFGS-B", prior: int = 1, distribe_shots: bool = True):
        """
        Run circuits with optimal shot distribution and optimal cut parameters.
        run_mode = "qasm": classical simulator
        num_shots_prior: number of prior shots for each subcircuit
        num_shots_total: total number of shots
        num_iter: total iteration rounds
        """
        # Preparation Stage
        self.prep_states = prep_states
        self.prior = prior
        self.total_shots_run = 0
        self.num_shots_given = num_shots_total

        # Generate entries
        print(f"--> Generating subcircuit entries:")
        self._generate_subcircuit_entries()
        for i in range(self.info["num_subcircuits"]):
            print(f"Subcircuit {i} has {self.info["num_subcircuits_entries"][i]} entries.")
        assert num_shots_prior*self.info["num_total_entries"] <= num_shots_total, f"Total number of prior shots({num_shots_prior*self.info["num_total_entries"]}) cannot exceed total shots({num_shots_total})."
        
        # Begin execution
        self._initialize_shot_count()
        initialize_counts(self.tmp_data_folder, self.subcircuits, self.subcircuits_entries)
        # Runmode: baseline
        if not distribe_shots:
            self._run_prior_samples(floor(num_shots_total / self.info["num_total_entries"]))
            self._generate_zero_params()
            return
        # Runmode: statevector
        if run_mode == "sv":
            self._generate_zero_params()
            self._run_sv()
        # Runmode: normal
        else:
            # Runtime: running prior samples
            self._run_prior_samples(num_shots_prior)
            # Runtime: optimizae parameters
            if use_params:
                self._optimize_params(method=method, prior=prior)
            else:
                self._generate_zero_params()
            # Runtime: running posterior samples
            num_shots_per_iter = floor((num_shots_total - num_shots_prior*self.info["num_total_entries"]) / num_iter)
            for round in range(num_iter):
                if self.verbose:
                    print(f"--> Iteration {round +1}: Running Posterior Samples")
                self._run_posterior_samples(num_shots_per_iter)


    def variance(self):
        current_prob_with_prior = read_probs_with_prior(self.tmp_data_folder, 0)
        meta_info = pickle.load(open("%s/meta_info.pckl" % (self.tmp_data_folder), "rb"))
        entry_dict = meta_info["entry_dict"]
        return calculate_variance(self.shot_count, self.params, current_prob_with_prior, entry_dict, self.info, self.subcircuits_info, self.prep_states)

    def _optimize_params(self, method, prior):
        if self.verbose:
            print("--> Optimizing Parameters")
        opt_cost, self.params = optimize_params(self.tmp_data_folder, self.info, self.subcircuits_info, self.prep_states, prior, method)
        if self.verbose:
            print("Optimized cost: ", opt_cost)
            print("Theoretical minimum variance: ", (opt_cost**2 / self.num_shots_given))
    

    def reconstruct(self):
        if self.verbose:
            print("--> Building output probability")
        self.output_prob = postprocess(self.tmp_data_folder, self.info, self.subcircuits_info, self.prep_states, self.params)


    def _run_prior_samples(self, num_shots_prior: int):
        if self.verbose:
            print("--> Running Prior Samples")
        run_samples(
            self.subcircuits,
            self.subcircuits_entries,
            "qasm",
            "equal",
            num_shots_prior,
            self.tmp_data_folder
        )
        for subcircuit_idx in range(self.info["num_subcircuits"]):
            for entry_idx in range(len(self.subcircuits_entries[subcircuit_idx])):
                self.shot_count[subcircuit_idx][entry_idx] += num_shots_prior


    def _run_sv(self):
        if self.verbose:
            print("--> Running Statevector Simulation")
        run_samples(
            self.subcircuits,
            self.subcircuits_entries,
            "sv",
            "sv",
            None,
            self.tmp_data_folder
        )


    def _run_posterior_samples(self, total_samples):
        current_prob_with_prior = read_probs_with_prior(self.tmp_data_folder, self.prior)
        meta_info = pickle.load(open("%s/meta_info.pckl" % (self.tmp_data_folder), "rb"))
        entry_dict = meta_info["entry_dict"]
        distribution = generate_distribution(total_samples, params_matrix_to_list(self.params), current_prob_with_prior, entry_dict, self.info, self.subcircuits_info, self.prep_states)
        # print(distribution)
        run_samples(
            self.subcircuits,
            self.subcircuits_entries,
            "qasm",
            "distribute",
            distribution,
            self.tmp_data_folder
        )
        for subcircuit_idx in range(self.info["num_subcircuits"]):
            for entry_idx in range(len(self.subcircuits_entries[subcircuit_idx])):
                self.shot_count[subcircuit_idx][entry_idx] += distribution[subcircuit_idx][entry_idx]


    def _generate_zero_params(self):
        if self.prep_states == [0,2,4,5]:
            num_param = 8
        elif self.prep_states == range(6):
            num_param = 24
        else:
            raise Exception("Invalid prep states")
        params = [[0 for _ in range(num_param)] for cut_idx in range(self.info["num_cuts"])]
        self.params = params


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
                    subcircuit_entries.append((tuple(prep[:]), tuple(meas[:])))
            self.subcircuits_entries.append(subcircuit_entries)
            num_subcircuits_entries.append(len(subcircuit_entries))
            num_total_entries += len(subcircuit_entries)
        self.info["num_subcircuits_entries"] = num_subcircuits_entries
        self.info["num_total_entries"] = num_total_entries
        # print(self.subcircuits_entries)


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
                            self.info["cuts"][cut_label][1] = (index, qregs[0]._index)
                        elif qubit_evolved[qregs[0]._index] == 1:
                            assert self.info["cuts"][cut_label][0] == None, f"Repeat measurement for cut {cut_label}"
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
                            instr.params[0] = 'm'
                            subcircuit_counter["O"] += 1
                            self.info["cuts"][cut_label][0] = (index, qregs[0]._index)
                    else:
                        if qubit_evolved[qregs[0]._index] == 0:
                            qubit_evolved[qregs[0]._index] = 1
                            subcircuit_input[qregs[0]._index] = cut_label
                            instr.params[0] = 'p'
                            subcircuit_counter["rho"] += 1
                            self.info["cuts"][cut_label] = [None, (index, qregs[0]._index)]
                        elif qubit_evolved[qregs[0]._index] == 1:
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
                            instr.params[0] = 'm'
                            subcircuit_counter["O"] += 1
                            self.info["cuts"][cut_label] = [(index, qregs[0]._index), None]
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
            subcircuit_info['num_qubits'] = subcircuit.num_qubits
            self.subcircuits_info.append(subcircuit_info)
            num_qubits += subcircuit_counter["effective"]
        for key in self.info["cuts"].keys():
            assert self.info["cuts"][key][0] != None and self.info["cuts"][key][1] != None, f"Cut {key} does not form a pair"
        self.info["num_subcircuits"] = len(self.subcircuits)
        self.info["num_qubits"] = num_qubits
        self.info["num_cuts"] = len(self.info["cuts"])
        self.info["cut_index"] = {}
        for index, key in enumerate(list(self.info["cuts"].keys())):
            self.info["cut_index"][key] = index
    

    def _initialize_shot_count(self):
        self.shot_count = []
        for subcircuit_idx in range(self.info["num_subcircuits"]):
            entry_count = [0 for _ in range(len(self.subcircuits_entries[subcircuit_idx]))]
            self.shot_count.append(entry_count)


    def print_info(self):
        for circuit in self.subcircuits:
            print(circuit)
        print(self.subcircuits_info)
        print(self.info)


    def clean_data(self):
        subprocess.run(["rm", "-r", self.tmp_data_folder])
    
    