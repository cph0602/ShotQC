from qiskit import QuantumCircuit
import os, subprocess, random, pickle, itertools, torch, time
from typing import List
from math import floor
from time import perf_counter
from shotqc.executor import run_samples
from shotqc.helper import initialize_counts, auto_total_shot
from shotqc.optimizer import parallel_optimize_params_sgd, parallel_minimize_var, batch_optimize_params, batch_minimize_var
from shotqc.parallel_overhead_v2 import (
    parallel_cost_function, parallel_reconstruct, 
    parallel_variance, parallel_distribute
)
from shotqc.args import Args

class ShotQC():
    """
    Main class for shotqc
    """
    def __init__(self, subcircuits: List[QuantumCircuit] | None = None, name: str | None = None, 
                 verbose: bool = False, reset_files: bool = True, device=None):
        assert subcircuits != None, "Subcircuits cannot be empty"
        self.name = name
        self.subcircuits = subcircuits
        self.info = {}
        self._extract_info()
        self.verbose = verbose
        self.use_cut_params = True
        self.tmp_data_folder = f"shotqc/{name}_tmp_data"
        self.reset_files = reset_files
        self.device=device
        if os.path.exists(self.tmp_data_folder) and reset_files:
            subprocess.run(["rm", "-r", self.tmp_data_folder])
        if reset_files:
            os.makedirs(self.tmp_data_folder)


    def execute(self, num_shots_prior: int | None = None, num_shots_total: int | None = None, prior_ratio: float | None = None,
                num_iter: int = 1, prep_states: List[int] = range(6), run_mode: str = "qasm", 
                use_params: bool = True, method: str = "SGD", prior: int = 0, distribe_shots: bool = True, 
                debug: bool = False, batch_size: int = 1024, ext_ratio: float = 1):
        """
        Run circuits with optimal shot distribution and optimal cut parameters.
        run_mode = "qasm": classical simulator
        num_shots_prior: number of prior shots for each subcircuit
        num_shots_total: total number of shots
        num_shots_ratio: automatic mode, determine the prior-posterior ratio
        num_iter: total iteration rounds
        """
        # Preparation Stage
        self.prep_states = prep_states
        self.prior = prior
        self.total_shots_run = 0
        if prep_states == [0,2,4,5]:
            self.num_params = 8
        elif prep_states == range(6):
            self.num_params = 24

        # Generate entries
        print(f"--> Generating subcircuit entries:")
        self._generate_subcircuit_entries()
        for i in range(self.info["num_subcircuits"]):
            print(f"Subcircuit {i} has {self.info["num_subcircuits_entries"][i]} entries.")
        if num_shots_total != None:
            self.num_shots_given = num_shots_total
            assert num_shots_prior*self.info["num_total_entries"] <= self.num_shots_given, f"Total number of prior shots({num_shots_prior*self.info["num_total_entries"]}) cannot exceed total shots({self.num_shots_given})."
        else:
            assert prior_ratio != None, f"Prior ratio must be specified when num_shots_total is not specified."
            self.num_shots_given = auto_total_shot(self.subcircuits, self.subcircuits_info, 4) * ext_ratio
            self.shot_ratio = self.num_shots_given / auto_total_shot(self.subcircuits, self.subcircuits_info, len(prep_states))
            print(f"--> Total number of shots: {self.num_shots_given}")
        
        
        # Begin execution
        self._initialize_shot_count()
        if self.reset_files:
            initialize_counts(self.tmp_data_folder, self.subcircuits, self.subcircuits_entries)
        # DEBUGGING
        if debug:
            print("--> Running Debug Snippet")
            start_time = perf_counter()
            self._generate_zero_params()
            # # initialize_counts(self.tmp_data_folder, self.subcircuits, self.subcircuits_entries)
            # # self._run_prior_samples(floor(num_shots_total / self.info["num_total_entries"]))
            # args = Args(self)
            # params = torch.zeros(self.num_params*self.info["num_cuts"])
            # with torch.no_grad():
            #     total_entry_coef(params, args, batch_size)
            print("Time Elapsed: ", perf_counter()-start_time)
            return
        # Runmode: statevector
        if run_mode == "sv":
            self._generate_zero_params()
            self._run_sv()
        # Runmode: baseline
        elif not distribe_shots:
            if num_shots_total != None:
                self._run_prior_samples(self.num_shots_given / self.info["num_total_entries"])
            else:
                self._run_prior_ratio_samples(1.0 * self.shot_ratio)
            self._generate_zero_params()
            return
        # Runmode: normal
        else:
            if num_shots_total != None:
                # Runtime: running prior samples
                self._run_prior_samples(num_shots_prior)
                # Runtime: optimizae parameters
                if use_params:
                    self._optimize_params(method=method, prior=prior, batch_size=batch_size)
                else:
                    self._generate_zero_params()
                # Runtime: running posterior samples
                num_shots_per_iter = (self.num_shots_given - num_shots_prior*self.info["num_total_entries"]) / num_iter
                if num_shots_per_iter != 0:
                    for round in range(num_iter):
                        if self.verbose:
                            print(f"--> Iteration {round +1}: Running Posterior Samples")
                        self._run_posterior_samples(num_shots_per_iter, batch_size=batch_size)
            else:
                # Runtime: running prior samples
                self._run_prior_ratio_samples(prior_ratio * self.shot_ratio)
                # Runtime: optimizae parameters
                if use_params:
                    self._optimize_params(method=method, prior=prior, batch_size=batch_size)
                else:
                    self._generate_zero_params()
                # Runtime: running posterior samples
                num_shots_per_iter = (self.num_shots_given * (1-prior_ratio)) / num_iter
                if num_shots_per_iter != 0:
                    for round in range(num_iter):
                        if self.verbose:
                            print(f"--> Iteration {round +1}: Running Posterior Samples")
                        self._run_posterior_samples(num_shots_per_iter, batch_size=batch_size)


    def variance(self, batch_size=1024):
        args = Args(self, device=self.device)
        actual_shot = 0
        for subcircuit_idx in range(self.info["num_subcircuits"]):
            for entry_idx in range(len(self.subcircuits_entries[subcircuit_idx])):
                actual_shot += self.shot_count[subcircuit_idx][entry_idx]
        print(f"Actual shots run: {actual_shot}")
        # cost = parallel_cost_function(self.params, args, batch_size=batch_size).item()
        # print("Theoretical Min. Variance: ", cost**2/self.num_shots_given)
        with torch.no_grad():
            variance = parallel_variance(self.params, args, self.shot_count, device=self.device, batch_size=batch_size).item()
        return variance


    def _optimize_params(self, method, prior=1, init_params=None, batch_size=1024):
        if self.verbose:
            print("--> Optimizing Parameters")
        if init_params == None:
            init_params = torch.zeros(self.info["num_cuts"] * self.num_params, requires_grad=True)
        args = Args(self, prior=prior, device=self.device)
        opt_cost, self.params = batch_optimize_params(
            init_params=init_params,
            args=args,
            device=self.device,
            batch_size=batch_size
        )
        if self.verbose:
            print("Optimized cost: ", opt_cost)
            print("Theoretical minimum variance: ", (opt_cost**2 / self.num_shots_given))
    

    def reconstruct(self, final_optimize=False, batch_size=1024):
        args = Args(self, device=self.device)
        # print(parallel_variance(self.params, args, self.shot_count, batch_size=batch_size, device=self.device).item())
        if final_optimize:
            print("--> Optimizing final variance")
            final_var, self.params = batch_minimize_var(self.params, args, self.shot_count, batch_size=batch_size, device=self.device)
        if self.verbose:
            print("--> Building output probability")
        # self.output_prob = postprocess(self.tmp_data_folder, self.info, self.subcircuits_info, self.prep_states, torch.tensor(self.params))
        with torch.no_grad():
            # print(batch_size)
            parallel_reconstruct(self.params, args=args, batch_size=batch_size, device=self.device)


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
    
    def _run_prior_ratio_samples(self, prior_ratio: float):
        if self.verbose:
            print("--> Running Prior Samples")
        distribution = [[max(1024, 2**self.subcircuits_info[subcircuit_idx]["num_qubits"]) * prior_ratio for entry_idx in range(self.info["num_subcircuits_entries"][subcircuit_idx])] for subcircuit_idx in range(self.info["num_subcircuits"])]
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


    def _run_posterior_samples(self, total_samples, batch_size):
        args = Args(self, device=self.device)
        if self.verbose:
            print("-----> Distributing Shots")
        distribution = parallel_distribute(self.params, args=args, total_samples=total_samples, device=self.device, batch_size=batch_size)
        if self.verbose:
            print("-----> Running Samples")
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
        self.params = torch.zeros(self.info["num_cuts"] * self.num_params, requires_grad=True)


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


    def _extract_info(self):
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
                if instr.name == "qpd_1q":
                    cut_label = instr.label
                    assert cut_label not in subcircuit_cuts, "Same cut cannot appear in one subcircuit"
                    subcircuit_cuts.add(cut_label)
                    assert qubit_evolved[qregs[0]._index] != -1, "No gate/cut allowed after measurement cut"
                    if cut_label in self.info["cuts"].keys():
                        if qubit_evolved[qregs[0]._index] == 0:
                            assert self.info["cuts"][cut_label][1] == None, f"Repeat preparation for cut {cut_label}"
                            qubit_evolved[qregs[0]._index] = 1
                            subcircuit_input[qregs[0]._index] = cut_label
                            subcircuit_counter["rho"] += 1
                            self.info["cuts"][cut_label][1] = (index, qregs[0]._index)
                        elif qubit_evolved[qregs[0]._index] == 1:
                            assert self.info["cuts"][cut_label][0] == None, f"Repeat measurement for cut {cut_label}"
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
                            subcircuit_counter["O"] += 1
                            self.info["cuts"][cut_label][0] = (index, qregs[0]._index)
                    else:
                        if qubit_evolved[qregs[0]._index] == 0:
                            qubit_evolved[qregs[0]._index] = 1
                            subcircuit_input[qregs[0]._index] = cut_label
                            subcircuit_counter["rho"] += 1
                            self.info["cuts"][cut_label] = [None, (index, qregs[0]._index)]
                        elif qubit_evolved[qregs[0]._index] == 1:
                            qubit_evolved[qregs[0]._index] = -1
                            subcircuit_output[qregs[0]._index] = cut_label
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
    
    