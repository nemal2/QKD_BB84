"""Separated-device BB84 probabilities and distinct physical channel mechanisms."""
import numpy as np
from scipy.optimize import brentq
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import thermal_relaxation_error
from common import h2,entropy_benchmark
I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);H=np.array([[1,1],[1,-1]])/np.sqrt(2)
U={'id':I,'x':X,'h':H}
PREP={ (0,0):['id','id'],(1,0):['id','x'],(0,1):['id','h'],(1,1):['x','h'] }
PRIMARY={(0,0):[],(1,0):['x'],(0,1):['h'],(1,1):['x','h']}


def depol(rho,p):return (1-p)*rho+p*I/2


def gate(rho,g,p):return depol(U[g]@rho@U[g].conj().T,p)


def physical_transition(model,scale=1.):
    result=np.empty((2,2))
    for b in [0,1]:
        for bit in [0,1]:
            rho=np.array([[1.,0],[0,0]],complex)
            for owner,seq in [('Alice',PREP[bit,b]),('Bob',['h'] if b else ['id'])]:
                if owner=='Bob' and model=='pauli':
                    px,py,pz=np.array([.015,.005,.075])*scale
                    rho=(1-px-py-pz)*rho+px*X@rho@X+py*Y@rho@Y+ pz*Z@rho@Z
                for g in seq:
                    rho=U[g]@rho@U[g].conj().T
                    if model=='homogeneous':rho=depol(rho,.03*scale)
                    elif model=='gate_dependent':rho=depol(rho,{'x':.02,'h':.05,'id':.005}[g]*scale)
                    elif model=='thermal':
                        duration={'x':40.,'h':60.,'id':50.}[g]*scale
                        rho=DensityMatrix(rho).evolve(thermal_relaxation_error(1800.,2500.,duration).to_quantumchannel()).data
            probabilities=np.real(np.diag(rho))
            if model=='readout':
                e0,e1=.01*scale,.11*scale
                probabilities=np.array([probabilities[0]*(1-e0)+probabilities[1]*e1,
                                        probabilities[0]*e0+probabilities[1]*(1-e1)])
            result[b,bit]=probabilities[1-bit]
    return result


def matched_transition(model,q=.06):
    bound={'homogeneous':30.,'gate_dependent':19.,'pauli':10.,'thermal':100.,'readout':8.}[model]
    scale=brentq(lambda t:physical_transition(model,t).mean()-q,0,bound,xtol=1e-11)
    return physical_transition(model,scale),scale


def q_path(pA,pB,b,x):
    k=x+b
    return (1-(1-pA)**k*(1-pB)**b)/2


def q_noeve(pA,pB):
    pA,pB=np.asarray(pA),np.asarray(pB)
    return sum(q_path(pA,pB,b,x) for b in [0,1] for x in [0,1])/4


def q_intercept(pA,pB,pE):
    pA,pB,pE=np.asarray(pA),np.asarray(pB),np.asarray(pE);same=0.
    for b in [0,1]:
        for x in [0,1]:
            f=(1-(1-pA)**(x+b)*(1-pE)**b)/2
            g=(1-(1-pE)**(x+b)*(1-pB)**b)/2
            other=(1-(1-pE)**(1-x+b)*(1-pB)**b)/2
            same+=(1-f)*g+f*(1-other)
    return .25+same/8


def Q(f,pA,pB,pE,r,case='legitimate'):
    f=np.asarray(f)
    a=f*pA;b=f*pB;e=pE*f if case=='all_scaled' else pE
    return (1-r)*q_noeve(a,b)+r*q_intercept(a,b,e)


def exact_density_Q(f,pA,pB,pE,r,case='legitimate'):
    # Explicit quantum-state walk, independent of the closed-form path counts.
    a=f*pA;bb=f*pB;ee=pE*f if case=='all_scaled' else pE
    q0=qi=0.
    for b in [0,1]:
        for bit in [0,1]:
            rho=np.array([[1.,0],[0,0]],complex)
            for g in PRIMARY[bit,b]:rho=gate(rho,g,a)
            direct=gate(rho,'h',bb) if b else rho
            q0+=float(np.real(direct[1-bit,1-bit]))/4
            for eb in [0,1]:
                eve=gate(rho,'h',ee) if eb else rho
                pe=np.real(np.diag(eve))
                for v in [0,1]:
                    resend=np.array([[1.,0],[0,0]],complex)
                    for g in PRIMARY[v,eb]:resend=gate(resend,g,ee)
                    received=gate(resend,'h',bb) if b else resend
                    qi+=pe[v]*float(np.real(received[1-bit,1-bit]))/8
    return (1-r)*q0+r*qi
