"""Descriptive channel-information analysis of unchanged, retained experiments.

No shots, frames, decoder outcomes, or learned policies are generated here.
Entropies use the stored physical transition laws. Yield accounting uses paired
per-frame records. The diagnostic ambiguity is an exact response-model example.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
SUP = HERE.parent
RESULTS = SUP/'extensions/results'
SEEDS = [211,223,227,229,233]
MODELS = ['homogeneous','gate_dependent','pauli','thermal','readout']


def entropy(probabilities):
    values = np.asarray(probabilities, float)
    values = values[values > 0]
    return float(-np.sum(values*np.log2(values)))


def h2(p):
    return entropy([p,1-p])


def information(transitions):
    """Compute conditional entropies from the full P(Z,A,B), not a BSC proxy."""
    transitions = np.asarray(transitions,float)
    assert transitions.shape == (2,2) and np.all((transitions >= 0)&(transitions <= 1))
    joint = np.zeros((2,2,2))  # Z, A, B; A and Z are independent uniform bits.
    for z in [0,1]:
        for a in [0,1]:
            joint[z,a,a] = (1-transitions[z,a])/4
            joint[z,a,1-a] = transitions[z,a]/4
    assert np.isclose(joint.sum(),1)
    pooled = joint.sum(axis=0)
    hab = entropy(pooled)-entropy(pooled.sum(axis=0))
    habz = entropy(joint)-entropy(joint.sum(axis=1))
    # Independent binary-transition expression for each conditional entropy.
    def binary_conditional(u,v):
        return 1+(h2(u)+h2(v))/2-h2((1+u-v)/2)
    assert np.isclose(hab,binary_conditional(*transitions.mean(axis=0)),atol=1e-13)
    assert np.isclose(habz,np.mean([binary_conditional(*row) for row in transitions]),atol=1e-13)
    he = h2(float(transitions.mean()))
    hez = float(np.mean([h2(float(q)) for q in transitions.mean(axis=1)]))
    return dict(qber=float(transitions.mean()),H_A_given_B=hab,H_A_given_BZ=habz,
                G=max(0.,hab-habz),H_error=he,H_error_given_basis=hez,
                basis_prior_logloss_gain=max(0.,he-hez),
                bit_prior_logloss_gain=max(0.,hez-habz))


def ideal_response(f,p,r):
    f = np.asarray(f,float)
    attenuation = 1-f*p
    hardware = sum((1-attenuation**k)/2 for k in [0,1,2,3])/4
    return r/4+(1-r/2)*hardware


def basis_response(p,r):
    noise=np.array([p/4,(5*p-4*p*p+p**3)/4])
    return r/4+(1-r/2)*noise


def basis_inverse(rates):
    rect,diag=np.asarray(rates,float)
    contrast=(diag-rect)/(1-2*rect)
    p=1-np.sqrt(1-2*contrast)
    r=(4*rect-p)/(1-p/2)
    return np.array([p,r])


def density_basis_response(p,r):
    """Resolve each basis by explicit state propagation with ideal interception."""
    sys.path.insert(0,str(SUP/'extensions'))
    from physics import PRIMARY,gate
    rates=[]
    for basis in [0,1]:
        direct=intercepted=0.
        for bit in [0,1]:
            rho=np.array([[1.,0],[0,0]],complex)
            for operation in PRIMARY[bit,basis]:rho=gate(rho,operation,p)
            received=gate(rho,'h',p) if basis else rho
            direct+=float(np.real(received[1-bit,1-bit]))/2
            for eve_basis in [0,1]:
                measured=gate(rho,'h',0.) if eve_basis else rho
                for outcome in [0,1]:
                    resend=np.array([[1.,0],[0,0]],complex)
                    for operation in PRIMARY[outcome,eve_basis]:resend=gate(resend,operation,0.)
                    received=gate(resend,'h',p) if basis else resend
                    intercepted+=float(np.real(measured[outcome,outcome]*received[1-bit,1-bit]))/4
        rates.append((1-r)*direct+r*intercepted)
    return np.array(rates)


def analyse():
    inputs = set()
    def read_npz(path):
        inputs.add(path)
        with np.load(path,allow_pickle=False) as archive:
            return {key:archive[key] for key in archive.files}
    physical = []
    csvpath = RESULTS/'physical_results.csv';inputs.add(csvpath)
    with csvpath.open() as stream:
        stored = list(csv.DictReader(stream))
    for model in MODELS:
        for setting in ['fixed','matched']:
            reference = None
            for family in ['gallager','irregular']:
                for seed in SEEDS:
                    data = read_npz(RESULTS/f'physical_{family}_k{seed}_{model}_{setting}_frames.npz')
                    transitions = data['transitions']
                    if reference is None:reference=transitions
                    else:assert np.array_equal(transitions,reference)
                    info = information(transitions)
                    row = next(v for v in stored if v['model']==model and v['setting']==setting
                               and v['family']==family and int(v['matrix_seed'])==seed and v['method']=='selected')
                    assert np.isclose(info['G'],float(row['G']),atol=1e-13)
                    assert np.isclose(info['H_A_given_BZ'],np.mean(json.loads(row['conditional_entropy'])),atol=1e-13)
            physical.append(dict(model=model,setting=setting,transitions=reference.tolist(),**info))

    pipeline = []
    summarypath = RESULTS/'transfer_summary.json';inputs.add(summarypath)
    summaries = json.loads(summarypath.read_text())
    for family in ['gallager','irregular']:
        values = []
        for seed in SEEDS:
            tag=f'transfer_{family}_n16000_k{seed}_c2'
            pooled=read_npz(RESULTS/f'{tag}_pipeline_pooled.npz')
            selected=read_npz(RESULTS/f'{tag}_pipeline_selected.npz')
            n,s=16000,1000
            mp,ms=pooled['m'],selected['m']
            ip,is_=pooled['correct'],selected['correct']
            assert len(ip)==len(is_)==500
            vp,vs=(n-mp-64)/(n+s),(n-ms-64)/(n+s)
            disclosure=(mp-ms)/(n+s)
            recovery=vp*(1-ip.astype(int))-vs*(1-is_.astype(int))
            delta=vs*is_-vp*ip
            assert np.allclose(delta,disclosure+recovery,rtol=0,atol=2e-16)
            values.append(dict(seed=seed,pooled_row_fraction=float(np.mean(mp)/n),
                selected_row_fraction=float(np.mean(ms)/n),disclosure_pp=float(100*np.mean(disclosure)),
                recovery_pp=float(100*np.mean(recovery)),delta_pp=float(100*np.mean(delta))))
        means={key:float(np.mean([v[key] for v in values])) for key in values[0] if key!='seed'}
        target=next(v for v in summaries if v['family']==family and v['n']==16000 and v['condition']==2
                    and v['method']=='adaptive_pipeline')
        assert np.isclose(means['delta_pp'],100*target['mean'],atol=1e-12)
        half=float(student_t.ppf(.975,4)*np.std([v['delta_pp'] for v in values],ddof=1)/np.sqrt(5))
        assert np.allclose([means['delta_pp']-half,means['delta_pp']+half],np.array(target['ci'])*100,atol=1e-12)
        pipeline.append(dict(family=family,**means,matrix_values=values))
    q,a=.06,.75
    pooled_entropy=h2(q)
    conditional_entropy=(h2(q*(1-a))+h2(q*(1+a)))/2
    for row in pipeline:
        row['pooled_excess']=row['pooled_row_fraction']-pooled_entropy
        row['selected_excess']=row['selected_row_fraction']-conditional_entropy
        assert np.isclose(row['pooled_row_fraction']-row['selected_row_fraction'],
                          pooled_entropy-conditional_entropy+row['pooled_excess']-row['selected_excess'],atol=1e-13)

    # A nominal-grid state and a distinct, equally noisy single-setting state.
    p,r=.015,.1
    operating=float(ideal_response(1,p,r))
    p_noise=float(brentq(lambda x:ideal_response(1,x,0)-operating,0,.1,xtol=1e-14))
    assert np.isclose(ideal_response(1,p_noise,0),operating,atol=1e-13)
    sys.path.insert(0,str(SUP/'extensions'))
    from physics import exact_density_Q, Q
    density_errors=[]
    for pp,rr in [(p,r),(p_noise,0.)]:
        assert np.allclose(basis_response(pp,rr),density_basis_response(pp,rr),rtol=0,atol=1e-13)
        assert np.allclose(basis_inverse(basis_response(pp,rr)),[pp,rr],rtol=0,atol=1e-13)
        for scale in [0.,1.,2.,8.]:
            value=float(ideal_response(scale,pp,rr))
            assert np.isclose(value,float(Q(scale,pp,pp,0.,rr)),atol=1e-13)
            error=abs(value-exact_density_Q(scale,pp,pp,0.,rr))
            assert error<1e-12
            density_errors.append(error)
    inputs.update(SUP/'extensions'/name for name in ['physics.py','common.py'])
    return dict(status='passed',scope='Descriptive analysis of retained records and exact response laws; no new Monte Carlo experiments.',
        physical=physical,pipeline=pipeline,
        transfer_benchmark=dict(qber=q,asymmetry=a,n=16000,sample=1000,H_pooled=pooled_entropy,
            H_conditional=conditional_entropy,G=pooled_entropy-conditional_entropy,
            ideal_yield_difference_pp=100*16000/17000*(pooled_entropy-conditional_entropy)),
        aggregate_ambiguity=dict(attacked_p=p,attack_fraction=r,noise_only_p=p_noise,
            common_operating_qber=operating,residuals=[r/4,0.],maximum_density_check_error=max(density_errors),
            attacked_basis_rates=basis_response(p,r).tolist(),noise_only_basis_rates=basis_response(p_noise,0.).tolist(),
            single_setting_basis_inverse_verified=True),
        input_sha256={str(path.relative_to(SUP)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();report=analyse();target=HERE/'channel-information.json'
    rendered=json.dumps(report,indent=2)+'\n'
    if args.check:assert target.read_text()==rendered,'Stale channel-information analysis'
    else:target.write_text(rendered)
    print(json.dumps({key:value for key,value in report.items() if key not in ['physical','pipeline','input_sha256']},indent=2))


if __name__=='__main__':main()
