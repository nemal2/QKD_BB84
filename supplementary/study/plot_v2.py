"""Consistent vector figures from canonical records and actual source circuits."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reconciliation_v2 import OUT,P,COUNTS,qk,physical_parameter
from circuits_v2 import circuit

FIG=Path(__file__).resolve().parents[2]/'overleaf/images';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif',
 'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':9,
 'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
C=['#28638a','#b15f29','#4d8062','#865b8c']


def save(fig,name):fig.savefig(FIG/name,pad_inches=.08);plt.close(fig)


def circuits():
    fig,axs=plt.subplots(4,2,figsize=(9,5.7),layout='constrained')
    for si,scheme in enumerate(COUNTS):
        for b in [0,1]:
            ax=axs[si,b];ax.set_xlim(0,1.15);ax.set_ylim(0,1);ax.axis('off')
            if si==0:ax.set_title(['Rectilinear paths','Diagonal paths'][b],pad=12)
            if b==0:ax.text(-.1,.5,scheme,ha='right',va='center',fontweight='bold')
            ax.axvline(.77,color='.65',ls=':',lw=.8)
            for bit,y in [(0,.68),(1,.22)]:
                seq=[x.operation.name for x in circuit(bit,b,b,scheme).data if x.operation.name in ['id','x','h']]
                bob=1 if b or scheme=='B+' else 0;prep=seq[:-1] if bob else seq
                ax.plot([.1,1.02],[y,y],color='.3',lw=.8)
                ax.text(.04,y,r'$|0\rangle$',ha='center',va='center',fontsize=9)
                ax.text(.12,y+.12,f'bit {bit}',fontsize=8)
                for j,g in enumerate(prep):
                    x=.3+j*.12;ax.add_patch(Rectangle((x-.04,y-.09),.08,.18,facecolor='#e6edf2',edgecolor=C[0],lw=.8))
                    ax.text(x,y,'I' if g=='id' else g.upper(),ha='center',va='center',fontsize=9)
                if bob:
                    ax.add_patch(Rectangle((.82,y-.09),.08,.18,facecolor='#f6ece4',edgecolor=C[1],lw=.8))
                    ax.text(.86,y,'I' if seq[-1]=='id' else 'H',ha='center',va='center',fontsize=9)
                ax.text(1.04,y,'M',ha='center',va='center',fontsize=9)
                ax.text(1.12,y,f'{len(seq)}',ha='left',va='center',fontsize=9)
            if si==3:
                ax.text(.49,-.12,'Alice',ha='center',fontsize=9);ax.text(.9,-.12,'Bob',ha='center',fontsize=9)
    save(fig,'complete_circuits.pdf')


def exposure():
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.1),layout='constrained')
    for si,scheme in enumerate(COUNTS):
        for q in P['qber_targets']:
            d=np.load(OUT/f'main_{scheme.replace("+","plus")}_{round(q*100):02d}_frames.npz')
            pred=qk(float(d['physical_p']),COUNTS[scheme]).mean(axis=1)
            for b,marker in [(0,'o'),(1,'s')]:
                e=d['errors'][d['bases']==b];n=len(e);ph=e.mean();z=1.95996398454
                den=1+z*z/n;center=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
                x=q*100+(si-1.5)*.12
                axs[0].errorbar(x,100*ph,yerr=[[100*(ph-center+half)],[100*(center+half-ph)]],fmt=marker,color=C[si],ms=3.5,capsize=2)
                axs[0].plot(x,100*pred[b],marker='_',color=C[si],ms=9)
        c=np.linspace(0,.1,100);p=np.array([physical_parameter(scheme,.1,t) for t in c])
        qr=c+(1-2*c)*qk(p[:,None,None],COUNTS[scheme]).mean(axis=2)[:,0]
        qd=c+(1-2*c)*qk(p[:,None,None],COUNTS[scheme]).mean(axis=2)[:,1]
        axs[1].plot(100*c,qd/qr,color=C[si],label=scheme,lw=1.5)
    axs[0].set(xlabel='Target mean QBER (%)',ylabel='Basis QBER (%)',xticks=[2,6,10])
    axs[1].set(xlabel='Independent channel QBER (%)',ylabel='Diagonal / rectilinear QBER',xticks=[0,2.5,5,7.5,10])
    axs[1].legend(frameon=False,ncol=2);axs[1].axhline(1,color='.6',lw=.6)
    for ax in axs:ax.grid(axis='y',alpha=.15)
    save(fig,'gate_exposure.pdf')


def curves():
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.3),layout='constrained')
    for ax,scheme in zip(axs,['A','B+']):
        ds=[np.load(OUT/f'curve_{scheme.replace("+","plus")}_{round(q*100):02d}.npz') for q in P['curves']['qber_grid']]
        for j,m in enumerate([225,450,650]):
            for k,style,label in [(0,'-','pooled'),(1,'--','shrink')]:
                y=np.array([1-d['correct'][j,k].mean() for d in ds])*100
                ax.plot(100*np.array(P['curves']['qber_grid']),y,style,color=C[j],lw=1.4,label=f'm={m}, {label}')
        ax.set_title(scheme,loc='left',fontweight='bold');ax.axhline(1,color='.4',lw=.7,ls=':')
        ax.set(xlabel='Target QBER (%)',ylabel='FER (%)',ylim=(-.1,102),xticks=[2,4,6,8,10,12])
        ax.set_yscale('symlog',linthresh=1);ax.set_yticks([0,1,10,100],['0','1','10','100']);ax.grid(axis='y',alpha=.15)
    fig.legend(*axs[1].get_legend_handles_labels(),frameon=False,ncol=3,loc='outside lower center',fontsize=8)
    save(fig,'fixed_code_curves.pdf')


def operating():
    S=json.loads((OUT/'summary.json').read_text());fig,axs=plt.subplots(1,2,figsize=(8.8,3.2),layout='constrained')
    for si,scheme in enumerate(COUNTS):
        rows=[r for r in S['main'] if r['scheme']==scheme and r['decoder']=='BP' and r['comparison']=='operating']
        x=np.array([r['target_qber']*100 for r in rows])+(si-1.5)*.15;y=np.array([r['delta_yield']*100 for r in rows]);ci=np.array([r['delta_yield_ci'] for r in rows]).T*100
        axs[0].errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[0],ci[1]-y])),fmt=['o','s','^','D'][si],color=C[si],capsize=2,label=scheme)
        axs[0].set(xlabel='Target QBER (%)',ylabel='Shrink minus pooled yield (pp)',xticks=[2,6,10])
    for si,scheme in enumerate(['PRIMARY','A']):
        rows=[r for r in S['channel_fixed'] if r['scheme']==scheme and r['decoder']=='BP']
        x=np.array([r['channel_qber']*100 for r in rows]);y=np.array([r['delta_yield']*100 for r in rows]);ci=np.array([r['delta_yield_ci'] for r in rows]).T*100
        axs[1].errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[0],ci[1]-y])),fmt=['o-','s-'][si],color=C[si],capsize=2,label=scheme)
    for ax in axs:ax.axhline(0,color='.5',ls=':',lw=.8);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    axs[1].set(xlabel='Independent channel QBER (%)',ylabel='Pooled minus shrinkage FER (pp)',xticks=[0,2.5,5,7.5,10])
    save(fig,'operating_points.pdf')


def diagnostic():
    S=json.loads((OUT/'summary.json').read_text());fig,axs=plt.subplots(1,2,figsize=(8.8,3.3),layout='constrained')
    labels={'linear_four':'Linear, four settings','gate_four':'Gate model, four settings','gate_two':'Gate model, two settings','calibrated_gate':'Calibrated gate'}
    for j,(key,label) in enumerate(labels.items()):
        rows=[x for x in S['diagnostic_by_attack'] if x['method']==key]
        axs[0].plot([x['p_eve'] for x in rows],[x['mae_pp'] for x in rows],['o-','s-','^-','D-'][j],ms=3,color=C[j],label=label)
    for j,key in enumerate(['calibrated_gate','Fiorini','calibrated_BSC']):
        rows=[x for x in S['drift'] if x['method']==key];x=100*np.array([r['delta'] for r in rows]);y=[r['mae_pp'] for r in rows]
        axs[1].plot(x,y,['o-','s--','^:'][j],color=C[j],ms=3,label=['Calibrated gate','Fiorini','Calibrated BSC'][j])
    two=next(r for r in S['diagnostic'] if r['method']=='gate_two')
    axs[1].axhline(two['mae_pp'],color=C[3],lw=1.4,label='Two-setting gate model')
    axs[1].axvspan(-20,20,color='.7',alpha=.12)
    axs[0].set(xlabel=r'Interception probability $p_{\mathrm{eve}}$',ylabel='MAE (pp)',xticks=[0,.1,.2,.3,.4])
    axs[1].set(xlabel='Calibration gate-noise offset (%)',ylabel='Pooled MAE (pp)',xticks=[-50,-20,0,20,50])
    for ax in axs:ax.legend(frameon=False,fontsize=7.3);ax.grid(axis='y',alpha=.15)
    save(fig,'diagnostic_design.pdf')


if __name__=='__main__':circuits();exposure();curves();operating();diagnostic()
