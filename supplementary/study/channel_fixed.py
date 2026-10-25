"""Fixed-disclosure channel sensitivity; no rate tuning on channel outcomes."""
import json
import numpy as np
from reconciliation_v2 import OUT,P,build_library,make_frames,evaluate,set_num_threads,get_num_threads


def main():
    set_num_threads(min(6,get_num_threads()));codes=build_library(11)
    selections=json.loads((OUT/'selections.json').read_text())
    for scheme in ['PRIMARY','A']:
        m=selections[f'{scheme}_0.1']['selected_m'][1]
        for channel in [.025,.05,.075,.10]:
            tag=f'channel_{scheme}_{round(channel*1000):03d}'
            path=OUT/f'chfixed_{scheme}_{round(channel*1000):03d}_evaluation.npz'
            if path.exists():continue
            d=np.load(OUT/f'{tag}_frames.npz')
            data=make_frames(scheme,.1,len(d['bits']),int(d['seed']),channel)
            for key in d.files:assert np.array_equal(d[key],np.asarray(data[key])),(tag,key)
            res=evaluate(codes[m],data,which=(0,3),strong=True,patterns=True)
            np.savez_compressed(path,m_values=[m],
                correct=[[r['correct'] for r in res]],bp_correct=[[r['bp_correct'] for r in res]],
                iterations=[[r['iterations'] for r in res]],retried=[[r['retried'] for r in res]],
                estimated_errors=[[r['estimated_errors'] for r in res]],collisions=[[r['collisions'] for r in res]])
            print(tag,m,[int((~r['bp_correct']).sum()) for r in res],flush=True)


if __name__=='__main__':main()
