"""
bb84_core.py
============
Quantum parties and classical post-processing for BB84 QKD.
 
Exports
-------
Alice            qubit preparation
Bob              qubit measurement
Eve              intercept-resend attack  (Phase 1)
QuantumChannel   noise model + Qiskit backend
sift_keys()      basis reconciliation
estimate_qber()  QBER with Wilson confidence interval
 
Future phases extend these classes, not this import surface.
"""


from __future__ import annotations
 
import random
from typing import Dict, List, Optional
 
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
 
#from bb84_config import QBERResult

class QuantumChannel:
    def __init__(
        self,
        noise_enabled: bool  = False,
        depolar_prob:  float = 0.0,
    ):
        self.noise_enabled = noise_enabled
        self.depolar_prob  = depolar_prob
        self._simulator    = self._build_simulator()