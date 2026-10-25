"""Fill evidence tables into the manuscript; keep the Overleaf paper in main.tex."""
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]
S=json.loads((OUT/'summary.json').read_text())


def ci(v,d=2,scale=1,signed=False):
    f=f'{{:{"+" if signed else ""}.{d}f}}'
    return '['+', '.join(f.format(x*scale) for x in v)+']'


def fer(r,key):
    return f"{r['failures_'+key]}: {100*r['fer_'+key]:.2f} {ci(r['fer_'+key+'_ci'],2,100)}"


def table(label,caption,columns,head,rows):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\small\setlength{\tabcolsep}{3.2pt}',
        '\\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+columns+r'@{}}',
        r'\toprule',head+r' \\',r'\midrule',*[r' & '.join(row)+r' \\' for row in rows],
        r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def main():
    op=[r for r in S['main'] if r['decoder']=='BP' and r['comparison']=='operating']
    operating=table('tab:operating',
      'Main BP operating policies. Each cell has 2000 independent holdout frames. FER entries give failure counts, percentages, and Wilson 95\\% intervals. U denotes pooled priors and A centred shrinkage. $m$ and $f_{\\mathrm{EC}}$ are shown as U/A pairs. Yield differences and paired 95\\% intervals are in percentage points. Target QBER labels denote generator expectations.',
      'lcccllc',r'Encoding & $Q$ (\%) & $m$ U/A & $f_{\mathrm{EC}}$ U/A & FER U (\%) & FER A (\%) & $\Delta Y$ (pp)',
      [[r['scheme'],f"{100*r['target_qber']:.0f}",f"{r['m_pooled']}/{r['m_aware']}",f"{r['f_ec_pooled']:.2f}/{r['f_ec_aware']:.2f}",
        fer(r,'pooled'),fer(r,'aware'),f"{100*r['delta_yield']:+.2f} {ci(r['delta_yield_ci'],2,100,True)}"] for r in op])
    matched=[r for r in S['main'] if r['target_qber']==.1 and r['comparison']=='same_aware_code']
    matched_table=table('tab:matched',
      'Matched-code comparisons at 10\\% target QBER. Both priors use the shrinkage-selected matrix and the same 2000 frames. Counts accompany absolute FER; R/H are rescued/harmed pairs. Ratios include paired bootstrap 95\\% intervals when at least five pooled failures support estimation; a dash indicates an undefined or too-sparse ratio interval. Holm correction covers the 12 main cells separately for BP and BP+OSD.',
      'llcllccc',r'Encoding & Decoder & $m$ & FER U (\%) & FER A (\%) & R/H & Ratio [95\% CI] & $p_{\rm Holm}$',
      [[r['scheme'],r['decoder'],str(r['m_aware']),fer(r,'pooled'),fer(r,'aware'),f"{r['rescued']}/{r['harmed']}",
        ('--' if r['ratio'] is None else f"{r['ratio']:.3f}")+((' '+ci(r['ratio_ci'])) if r['ratio_ci'] else ' [--]'),
        f"{r['p_holm']:.2g}"] for r in matched])
    names={'linear_four':'Linear, four settings','gate_four':'Gate model, four settings','gate_two':'Gate model, two settings',
           'calibrated_BSC':'Calibrated BSC','calibrated_gate':'Calibrated gate','Fiorini':'Fiorini formula'}
    diag_table=table('tab:diagnostic',
      'Equal-budget diagnostic performance at zero calibration drift. Every method uses 8000 transmitted signals per replicate; 1000 replicates at each of five attack fractions are pooled with equal weight. Intervals are stratified bootstrap 95\\% intervals for MAE. Signed bias and root-mean within-target variance (SD) distinguish bias from sampling dispersion. All errors are in percentage points and estimates are unclipped.',
      'lccc',r'Estimator & MAE [95\% CI] (pp) & Signed bias (pp) & SD (pp)',
      [[names[r['method']],f"{r['mae_pp']:.3f} {ci(r['mae_ci_pp'],3)}",f"{r['bias_pp']:+.3f}",f"{r['sd_pp']:.3f}"] for r in S['diagnostic']])
    a=next(r for r in op if r['scheme']=='A' and r['target_qber']==.1)
    b=next(r for r in op if r['scheme']=='B+' and r['target_qber']==.1)
    diag={r['method']:r for r in S['diagnostic']}
    ab=[r for r in S['ablation'] if r['scheme']=='B+' and r['target_qber']==.1 and r['decoder']=='BP']
    vals={r['prior']:r['failures'] for r in ab}
    replacements={'METADATA':json.loads((HERE.parent/'manuscript_support/frontmatter.json').read_text())['metadata'],
       'OPERATING_TABLE':operating,'MATCHED_TABLE':matched_table,'DIAGNOSTIC_TABLE':diag_table,
       'A10_MU':str(a['m_pooled']),'A10_MA':str(a['m_aware']),'A10_FU':f"{a['f_ec_pooled']:.3f}",'A10_FA':f"{a['f_ec_aware']:.3f}",
       'A10_Y':f"{100*a['delta_yield']:+.2f}",'A10_YCI':ci(a['delta_yield_ci'],2,100,True),
       'B10_Y':f"{100*b['delta_yield']:+.2f}",
       'ABLATION_TEXT':f"At the balanced B+ matched code at 10\\% QBER, BP fails on {vals['pooled']}, {vals['separate_Jeffreys']}, {vals['coherent_unshrunk']}, and {vals['coherent_shrink']} of 2000 frames for pooled, separate-Jeffreys, coherent unshrunk, and shrinkage priors, respectively. These sparse counts provide limited discrimination among estimators; their direction nevertheless cautions against assigning noisy class contrasts automatically.",
       'TWO_MAE':f"{diag['gate_two']['mae_pp']:.3f}",'FOUR_MAE':f"{diag['gate_four']['mae_pp']:.3f}",'CAL_MAE':f"{diag['calibrated_gate']['mae_pp']:.3f}",
       'DIAG_DIFFCI':ci(S['two_point_minus_calibrated_mae_ci_pp'],3,signed=True),
       'DRIFT_TEXT':'For the optimized designs, the two-setting fit already has slightly lower pooled MAE at zero drift, so there is no crossover within the tested offsets for calibrated gate inversion or the Fiorini formula. Their errors rise as the reference moves away from the signal condition. The calibrated BSC approximation has an asymmetric drift curve because drift can partly cancel its model bias.'}
    body=(HERE/'paper_body.template.tex').read_text()
    for key,value in replacements.items():body=body.replace('@'+key+'@',value)
    assert not re.search(r'@[A-Z_0-9]+@',body)
    front=json.loads((HERE.parent/'manuscript_support/frontmatter.json').read_text())['head']
    front=re.sub(r'% REVISION\([^\n]+\):[^\n]*\n','',front)
    front=front.replace(r'\usepackage{placeins}',r'\usepackage{placeins}'+'\n'+
                        '% Allow small spacing adjustments before extending a line into the margin.\n'+r'\setlength{\emergencystretch}{2em}')
    (ROOT/'overleaf/main.tex').write_text(front+body)
    appendix=r'''\appendix
\section{Operating-Point and Estimator Details}
\label{app:ablation}
% Keep auxiliary evidence from the circuit experiment together.
Table~\ref{tab:ablation} compares all four estimators at the same shrinkage-selected
matrix in the 10\% cells. Table~\ref{tab:replication} transfers the frozen row counts to
two fresh matrices and independent frames. Each comparison therefore isolates estimator
or matrix variation without using the evaluation frame to retune the rate.
'''
    rows=[]
    for scheme in ['PRIMARY','A','B','B+']:
        for decoder in ['BP','BP+OSD']:
            rs=[r for r in S['ablation'] if r['scheme']==scheme and r['target_qber']==.1 and r['decoder']==decoder]
            rows.append([scheme,decoder,str(rs[0]['m_aware'])]+[str(r['failures']) for r in rs])
    appendix+=table('tab:ablation','Prior-estimation ablation at 10\\% QBER. Entries are exact failures out of 2000 paired frames on one common matrix in each cell. P is pooled Jeffreys; J is separate Jeffreys; C is coherent unshrunk; S is centred shrinkage. Complete counts at 2\\% and 6\\% accompany the same analysis.','llccccc',
      r'Encoding & Decoder & $m$ & P & J & C & S',rows)
    rs=[r for r in S['replication'] if r['comparison']=='same_aware_code']
    rows=[]
    for r in rs:
        if r['decoder']!='BP':continue
        strong=next(t for t in rs if t['code_seed']==r['code_seed'] and t['scheme']==r['scheme'] and t['decoder']=='BP+OSD')
        rows.append([str(r['code_seed']),r['scheme'],str(r['m_aware']),fer(r,'pooled'),fer(r,'aware'),fer(strong,'pooled'),fer(strong,'aware')])
    appendix+=table('tab:replication','Additional matrix seeds at 10\\% target QBER. Each seed/encoding uses 1000 fresh frames. Both priors use the shrinkage row count from independent seed-11 calibration. Entries show failure counts, FER percentages, and Wilson 95\\% intervals. OSD denotes BP+OSD.','clcllll',r'Seed & Encoding & $m$ & BP U (\%) & BP A (\%) & OSD U (\%) & OSD A (\%)',rows)
    appendix+=r'''
\FloatBarrier
\section{Two-Hop Diagnostic Model and Numerical Fitting}
\label{app:diagnostic}
For basis $b$ and bit $x$, let $k_{bx}$ be PRIMARY's complete noisy-gate count and
$a_{bx}=q_{k_{bx}}(p)$. Conditional on Eve choosing the same basis, her noisy result
changes which preparation is resent. Averaging over both possibilities gives
\begin{equation}
 q_I(p)=\frac14+\frac18\sum_{b=0}^1\sum_{x=0}^1
 \left[(1-a_{bx})a_{bx}+a_{bx}(1-a_{b,1-x})\right].
 \label{eq:twohop}
\end{equation}
The first term comes from Eve's wrong basis. The second includes both noisy hops.
Together with \eqref{eq:attack}, this yields the exact gate-model estimator.

For counts $E_j$ among $N_j$ sifted signals, the fits use observations $E_j/N_j$ and
weights $N_j/[u_j(1-u_j)]$, where $u_j=(E_j+1/2)/(N_j+1)$. Nonlinear weighted least
squares starts at $p=0.03$ and four times the linear intercept, with the starting
$r$ clipped to $[-0.4,1.4]$. Bounds are $10^{-8}\le p\le\min(0.3,0.999/\max f)$
and $-0.5\le r\le1.5$; numerical tolerances are $10^{-10}$. The extended fitted
range preserves signed errors near $r=0$, while physical simulated attacks remain
in $[0,1]$. Calibrated inversion uses the monotone $q_0$ map.

With $J_j=\nabla_{(p,r)}Q(f_jp,r)$ and expected sifted allocation $N_j=T_j/2$,
the Fisher matrix is $F=\sum_j N_jJ_jJ_j^{\mathsf T}/[Q_j(1-Q_j)]$.
The minimax objective is the worst $(F^{-1})_{rr}/16$ on the design grid.
Calibration allocation uses the delta-method variance for \eqref{eq:calgate}.

The Fiorini comparator follows the formula in the public source file
\texttt{bb84/bb84\_algorithm.py}; its commit identifier and MIT notice accompany
the implementation. In the native-law control, its mean signed errors over the
five attack fractions range from @NATIVE_LOW@ to @NATIVE_HIGH@ percentage points.
This checks the implementation separately from the change in physical noise model.
'''
    low=min(r['bias_pp'] for r in S['native_comparator']);high=max(r['bias_pp'] for r in S['native_comparator'])
    appendix=appendix.replace('@NATIVE_LOW@',f'{low:+.3f}').replace('@NATIVE_HIGH@',f'{high:+.3f}')
    appendix+=r'''
\section{Independent Channel Controls}
The reconciliation controls replace gate disturbance with an independent isotropic link
at fixed 10\% total QBER. Table~\ref{tab:channel} gives absolute BP failure counts
at the fixed gate-only calibrated row count. Separate adaptively calibrated channel
operating results accompany the data. The diagnostic control adds an unscaled receiver
channel after the entire intercepted or unintercepted path. Its exact zero-gate-noise
intercept is $c+(1-2c)r/4$. A two-parameter fit that assumes $c=0$ cannot separate
the contributions of unknown $c$ and $r$ from two settings. Retained simulations at
$c=1\%$ and $3\%$, and $r=0,0.2,0.4$, quantify this model mismatch.
'''
    rs=[r for r in S['channel_fixed'] if r['decoder']=='BP']
    appendix+=table('tab:channel','Fixed-code channel-composition BP controls at 10\\% total target QBER. Each cell has 2000 independent holdout frames. The zero-channel points reuse the main cohort. FER entries show counts, percentages, and Wilson 95\\% intervals.','lcc ll',r'Encoding & $c$ (\%) & $m$ & FER U (\%) & FER A (\%)',
      [[r['scheme'],f"{100*r['channel_qber']:.1f}",str(r['m']),fer(r,'pooled'),fer(r,'aware')] for r in rs])
    (ROOT/'overleaf/appendix.tex').write_text(appendix)
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',body,re.S)[1]
    words=len(abstract.split());assert words<200,words
    (OUT/'manuscript_checks.json').write_text(json.dumps(dict(abstract_words=words,unresolved_tokens=False),indent=2)+'\n')
    print('manuscript written; abstract words:',words)


if __name__=='__main__':main()
