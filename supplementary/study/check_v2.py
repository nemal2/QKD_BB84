"""Audit every retained decoded pattern and the calibration/holdout separation."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from reconciliation_v2 import OUT,P,build_library,wilson_upper


def main():
    decoded=partitions=collisions=0;seeds=[];codecache={}
    for path in sorted(OUT.glob('*_frames.npz')):
        d=np.load(path);seeds.append(int(d['seed']));n=len(d['bits']);partitions+=n
        ix=np.sort(np.column_stack([d['sample_indices'],d['kept_indices']]),axis=1)
        assert np.array_equal(ix,np.broadcast_to(np.arange(1200),(n,1200))),path.name
        e=np.take_along_axis(d['errors'],d['sample_indices'],axis=1)
        b=np.take_along_axis(d['bases'],d['sample_indices'],axis=1)
        expected=np.column_stack([(e*(b==0)).sum(1),(b==0).sum(1),(e*(b==1)).sum(1),(b==1).sum(1)])
        assert np.array_equal(expected,d['sample_counts']),path.name
        ep=path.with_name(path.name.replace('_frames','_evaluation'))
        if not ep.exists():continue
        ev=np.load(ep);true=np.take_along_axis(d['errors'],d['kept_indices'],axis=1)
        code_seed=int(path.name.split('_')[1]) if path.name.startswith('rep_') else 11
        if code_seed not in codecache:codecache[code_seed]=build_library(code_seed)
        windows=np.lib.stride_tricks.sliding_window_view(d['hash_seeds'],64,axis=1)[:,::-1]
        columns=np.ascontiguousarray(np.packbits(windows,axis=2,bitorder='little')).view('<u8').reshape(n,1000)
        for i,m in enumerate(ev['m_values']):
            H=csr_matrix(codecache[code_seed][int(m)])
            for j in range(4):
                estimate=ev['estimated_errors'][i,j];correct=np.all(estimate==true,axis=1)
                assert np.array_equal(correct,ev['correct'][i,j]),(path.name,m,j)
                diff=estimate^true
                assert np.all((H@diff.T).toarray()%2==0) if hasattr(H@diff.T,'toarray') else np.all((H@diff.T)%2==0)
                hv=np.bitwise_xor.reduce(np.where(diff,columns,np.uint64(0)),axis=1)
                collision=(hv==0)&~correct
                assert np.array_equal(collision,ev['collisions'][i,j]);collisions+=int(collision.sum());decoded+=n
    for path in sorted(OUT.glob('chfixed_*_evaluation.npz')):
        frame=path.with_name(path.name.replace('chfixed_','channel_').replace('_evaluation','_frames'))
        d=np.load(frame);ev=np.load(path);true=np.take_along_axis(d['errors'],d['kept_indices'],axis=1)
        H=csr_matrix(codecache[11][int(ev['m_values'][0])])
        windows=np.lib.stride_tricks.sliding_window_view(d['hash_seeds'],64,axis=1)[:,::-1]
        columns=np.ascontiguousarray(np.packbits(windows,axis=2,bitorder='little')).view('<u8').reshape(len(true),1000)
        for j in range(2):
            estimate=ev['estimated_errors'][0,j];correct=np.all(estimate==true,axis=1)
            assert np.array_equal(correct,ev['correct'][0,j]),(path.name,j)
            diff=estimate^true;assert np.all((H@diff.T)%2==0)
            hv=np.bitwise_xor.reduce(np.where(diff,columns,np.uint64(0)),axis=1)
            collision=(hv==0)&~correct;assert np.array_equal(collision,ev['collisions'][0,j])
            collisions+=int(collision.sum());decoded+=len(true)
    assert len(seeds)==len(set(seeds)),'Independent datasets reused a seed'
    calibration=[]
    for path in sorted(OUT.glob('*_selection.json')):
        d=json.loads(path.read_text());a=np.load(path.with_name(path.name.replace('_selection.json','_calibration.npz')))
        for j,m in enumerate(d['selected_m']):
            eligible=[]
            for value in a['m_values']:
                higher=a['correct'][a['m_values']>=value,j]
                if all(wilson_upper(int((~r).sum()),len(r))<=.01 for r in higher):eligible.append(int(value))
            assert m==min(eligible),(path.name,j,m,eligible)
        calibration.append(path.name)
    report=dict(status='passed',unique_frame_dataset_seeds=len(seeds),independent_sifted_frames=partitions,
                decoded_method_frames_checked=decoded,Toeplitz_collisions=collisions,
                calibration_selections_recomputed=len(calibration),all_final_patterns_syndrome_valid=True)
    (OUT/'record_integrity.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
