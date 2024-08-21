import argparse, pickle, subprocess
from shotqc.executor import modify_subcircuit
from qiskit import QuantumCircuit
import qiskit_aer as aer
import copy, psutil, os, itertools
import numpy as np


circuit = QuantumCircuit(4)
circuit.h(0)
circuit.cx(1,2)
circuit.cx(0,1)
circuit.cx(1,2)
circuit.h(1)
circuit.x(1)
circuit.cx(2,3)
print(circuit)
simulator = aer.Aer.get_backend('aer_simulator')
circuit.measure_all()
sim_job = simulator.run(circuit, shots = 1000)
sim_result = sim_job.result()
sim_counts = sim_result.get_counts(circuit)
print(sim_counts)