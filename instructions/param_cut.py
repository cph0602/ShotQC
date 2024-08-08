from qiskit.circuit import QuantumCircuit, Instruction
from math import pi
from enum import Enum

class Prep_Basis(Enum):
    PLUS_STATE = 0
    MINUS_STATE = 1
    PLUS_I_STATE = 2
    MINUS_I_STATE = 3
    ZERO_STATE = 4
    ONE_STATE = 5

class Meas_Basis(Enum):
    X = 0
    Y = 1
    Z = 2

class ParamCut(Instruction):
    """An instruction to denote the cut location for subciruits"""
    def __init__(self, label: str | None = None, mode: str = 'u', basis: int | Prep_Basis | Meas_Basis | None = None):
        """
        Param_Cut: Instruction for location/pair of cuts on subcircuits\n
        Allowed modes:
            'u': undefined (default)
            'm': measurement
            'p': state preparation
        Basis for each mode:
            'u': None
            'm': Meas_Basis or [0, 1, 2]
            'p': Prep_Basis or [0, 1, 2, 3, 4, 5]
        """
        assert mode in ['u', 'm', 'p'], "Invalid cut mode"
        if mode == 'm':
            assert (basis in Meas_Basis) or (basis in [0, 1, 2, None]), "Invalid basis for preparation"
        if mode == 'p':
            assert (basis in Prep_Basis) or (basis in [0, 1, 2, 3, 4, 5, None]), "Invalid basis for measurement"
        allowed_types = (int, type(None))
        if isinstance(basis, allowed_types):
            temp = basis
        else:
            temp = basis.value
        super().__init__("param_cut", 1, 0, [mode, temp])
        self.label = label
    
    def _define(self):
        circuit = QuantumCircuit(1, name=self.name)
        self.definition = circuit