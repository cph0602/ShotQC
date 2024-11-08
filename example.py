from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from shotqc.main import ShotQC
from helper_functions.compare import ground_truth, squared_error, vector_ground_truth
from helper_functions.ckt_cut import cut_circuit
from helper_functions.helper_instr import PseudoQPD1Q
from testbench.qaoa import qaoa_circuit
from math import pi
import numpy as np
import networkx as nx
from itertools import product, combinations
from testbench.adder import adder20, adder20_0, adder20_1
import torch, argparse, os

parser = argparse.ArgumentParser(description="Experiment argument parsers")
parser.add_argument("name", type=str, help="circuit name")
parser.add_argument("num_qubits",type=int, help="number of qubits")
args = parser.parse_args()
org_ckt_path = f"benchmarks/{args.name}/{args.name}_{args.num_qubits}.qasm"
sub0_path = f"benchmarks/{args.name}/{args.name}_{args.num_qubits}_subcircuit_0.qasm"
sub1_path = f"benchmarks/{args.name}/{args.name}_{args.num_qubits}_subcircuit_1.qasm"

org_ckt = QuantumCircuit.from_qasm_file(org_ckt_path)
sub0 = QuantumCircuit.from_qasm_file(sub0_path)
sub1 = QuantumCircuit.from_qasm_file(sub1_path)