"""Actual source circuits and exact probability calculations (no shot fitting)."""
import sys
from pathlib import Path
import numpy as np
from qiskit import qasm2
from qiskit_aer import AerSimulator
from qiskit_aer.utils import insert_noise
from qiskit_aer.noise import depolarizing_error
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'upstream'))
from bb84_core import Alice,Bob
from bb84_noise import QuantumChannel,NoiseModelType
from reconciliation_v2 import COUNTS,OUT


def circuit(bit,basis,measurement_basis,scheme='PRIMARY'):
    a=Alice(1,seed=0);a.bits[0]=bit;a.bases[0]=basis
    c=a.prepare_qubit(0,gate_scheme=None if scheme=='PRIMARY' else scheme.replace('+',''))
    if scheme=='B+' and measurement_basis==0:c.id(0)
    class Recorder:
        def run_circuit(self,qc,shot_seed=None):self.qc=qc.copy();return 0
    b=Bob(1,seed=0);b.bases[0]=measurement_basis;rec=Recorder();b.measure(c,0,rec)
    return rec.qc


def probabilities(c,p,channel=0):
    noise=QuantumChannel(noise_model=NoiseModelType.DEPOLARIZING,depolar_prob=p)
    z=insert_noise(c,noise._simulator.options.noise_model,transpile=False)
    z.remove_final_measurements()
    if channel:z.append(depolarizing_error(2*channel,1).to_instruction(),[0])
    z.save_probabilities()
    return np.asarray(AerSimulator(method='density_matrix').run(z,shots=1).result().data(0)['probabilities'])


def export():
    directory=OUT/'circuits';directory.mkdir(exist_ok=True)
    for scheme in COUNTS:
        for b in [0,1]:
            for x in [0,1]:
                c=circuit(x,b,b,scheme)
                assert sum(i.operation.name in ('id','x','h') for i in c.data)==COUNTS[scheme][b,x]
                (directory/f'{scheme.replace("+","plus")}_b{b}_x{x}.qasm').write_text(qasm2.dumps(c)+'\n')


if __name__=='__main__':export()
