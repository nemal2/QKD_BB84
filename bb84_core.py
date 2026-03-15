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


   #  Private helpers 
 
    def _build_simulator(self) -> AerSimulator:
        if self.noise_enabled and self.depolar_prob > 0:
            noise_model = NoiseModel()
            # Depolarizing channel: E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
            error = depolarizing_error(self.depolar_prob, 1)
            noise_model.add_all_qubit_quantum_error(error, ["x", "h", "id"])
            print(f"  [Channel] Depolarizing noise  p = {self.depolar_prob}")
            return AerSimulator(noise_model=noise_model)
        return AerSimulator()
 
    #  Public interface 
 
    def run_circuit(self, qc: QuantumCircuit) -> int:
        """
        Simulate a single-qubit measurement circuit.
        Returns the measured bit (0 or 1).
 
        Phase 3 note: photon-loss model will need a 'lost' return value
        (e.g. None); handle that in QuantumChannel subclass, not here.
        """
        job    = self._simulator.run(transpile(qc, self._simulator), shots=1)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])
    

# Implementation of Class Alice

class Alice:
    """
    Alice randomly selects bits and bases, then encodes each bit
    as a qubit.
 
    Encoding table
    ==============
    bit=0, basis=0  →  |0⟩        (no gates)
    bit=1, basis=0  →  |1⟩        (X gate)
    bit=0, basis=1  →  |+⟩        (H gate)
    bit=1, basis=1  →  |−⟩        (X → H)
 
    Bases
    ======
    0 = Rectilinear (+)
    1 = Diagonal    (x)
    """
 
    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n_qubits = n_qubits
        rng = np.random.default_rng(seed)
        self.bits  = rng.integers(0, 2, n_qubits).tolist()
        self.bases = rng.integers(0, 2, n_qubits).tolist()
 
    def prepare_qubit(self, index: int) -> QuantumCircuit:
        """Return a 1-qubit circuit encoding bits[index] in bases[index]."""
        qc = QuantumCircuit(1, 1, name=f"A{index}")
        if self.bits[index] == 1:
            qc.x(0)
        if self.bases[index] == 1:
            qc.h(0)
        return qc
 
    def sift_key(self, matching_indices: List[int]) -> List[int]:
        """Alice's bits at positions where bases agreed with Bob's."""
        return [self.bits[i] for i in matching_indices]