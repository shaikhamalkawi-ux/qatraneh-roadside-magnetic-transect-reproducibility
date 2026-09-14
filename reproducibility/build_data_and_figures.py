# This builder reproduces fixed-table figures and station-layout outputs.
# Qatraneh G/P/E model selection is separately recomputed by recompute_qatraneh_gep.py,
# which is the authoritative reproducibility path for nested model decisions and robustness checks.
from pathlib import Path
import itertools, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
matplotlib.rcParams.update({
    'font.size': 13,
    'font.weight': 'bold',
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.linewidth': 1.25,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})
import matplotlib.pyplot as plt

def style_axis(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_color('black')
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color('black')

from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIG=ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

dist=[1,2,3,4,5,6,7,8,9,10,12,14]
north={
'Field K':[6.8,4.6,10.6,7.0,21.6,41.0,58.6,40.2,64.6,78.0,64.25,62.4],
'Fine LF':[2.7,4.6,5.1,5.5,6.2,7.1,8.1,7.2,8.2,9.0,8.3,8.3],
'Fine HF':[2.9,4.6,4.7,5.5,5.8,6.9,7.9,7.0,8.1,8.6,8.1,7.9],
'Coarse LF':[0.74,1.54,1.90,1.92,2.60,4.50,6.90,2.32,6.70,7.70,7.80,8.00],
'Coarse HF':[0.74,1.54,1.90,1.92,2.50,4.30,6.80,2.25,6.30,7.40,7.50,7.70],
}
south={
'Field K':[2.8,5.6,25.5,23.2,37.6,54.6,48.5,59.0,50.6,56.8,57.0,59.4],
'Fine LF':[1.2,2.2,5.8,5.5,7.1,8.7,8.0,8.6,8.1,8.6,8.1,8.8],
'Fine HF':[1.2,2.2,5.8,5.2,6.9,8.0,8.0,8.1,8.1,8.0,7.8,8.8],
'Coarse LF':[0.65,1.20,3.45,2.70,7.60,8.50,8.10,8.30,7.90,7.50,7.50,7.20],
'Coarse HF':[0.55,1.10,3.45,2.60,7.40,8.10,7.70,8.10,7.50,7.00,7.20,7.00],
}

# Canonical observation table
rows=[]
for side, dd, prefix in [('North',north,'N'),('South',south,'S')]:
    for i,d in enumerate(dist):
        rows.append({'ID':f'{prefix}{d:02d}','Side':side,'Distance_m':d,
                     'Kfield_x1e-6_SI':dd['Field K'][i],
                     'Fine_LF_x1e-8_m3kg':dd['Fine LF'][i],
                     'Fine_HF_x1e-8_m3kg':dd['Fine HF'][i],
                     'Coarse_LF_x1e-8_m3kg':dd['Coarse LF'][i],
                     'Coarse_HF_x1e-8_m3kg':dd['Coarse HF'][i]})
obs=pd.DataFrame(rows)
obs.to_csv(DATA/'Qatraneh_R15R4_Observations.csv',index=False)

def ts(x,y):
    s=[]
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if x[j] != x[i]: s.append((y[j]-y[i])/(x[j]-x[i]))
    return float(np.median(s))

def spr(x,y):
    rx=rankdata(x, method='average'); ry=rankdata(y, method='average')
    return float(np.corrcoef(rx,ry)[0,1])

routes=[]
for side,dd in [('North',north),('South',south)]:
    for route, vals in dd.items():
        routes.append((side,route,np.asarray(dist,float),np.asarray(vals,float)))

# Cutpoint contrasts and LOO
cut=[]; loo=[]; summary=[]
for side,route,x,y in routes:
    for c in range(3,9):
        inn=y[x<=c]; out=y[x>c]
        cut.append({'Side':side,'Route':route,'Cutpoint_m':c,'n_inner':len(inn),'n_outer':len(out),
                    'Mean_inner':inn.mean(),'Mean_outer':out.mean(),'Outer_minus_inner':out.mean()-inn.mean(),
                    'Outer_vs_inner_pct':(out.mean()-inn.mean())/inn.mean()*100})
    for omit in range(len(x)):
        mask=np.arange(len(x))!=omit
        loo.append({'Side':side,'Route':route,'Omitted_distance_m':x[omit],
                    'Spearman_rho':spr(x[mask],y[mask]),'TheilSen_slope':ts(x[mask],y[mask])})
    summary.append({'Side':side,'Route':route,'Spearman_rho':spr(x,y),'TheilSen_slope':ts(x,y),
                    'C10':float(y[x<=10].max()/y.max()),'Outer_max':bool(np.argmax(y)==len(y)-1)})
pd.DataFrame(cut).to_csv(DATA/'Qatraneh_R15R4_Cutpoint_Contrasts.csv',index=False)
pd.DataFrame(loo).to_csv(DATA/'Qatraneh_R15R4_LeaveOneDistance.csv',index=False)

# Sparse subset enumeration
sparse=[]; kstars={}
for side,route,x,y in routes:
    all_R={}
    for k in range(2,13):
        pos=tot=zero=0
        for inds in itertools.combinations(range(12),k):
            tot+=1
            slope=ts(x[list(inds)],y[list(inds)])
            if slope>0: pos+=1
            elif slope==0: zero+=1
        if route=='Field K':
            assert zero == 0, 'Field negative-layout figure requires zero-free subsets'
        R=pos/tot
        all_R[k]=R
        sparse.append({'Side':side,'Route':route,'k':k,'positive':pos,'total':tot,'R_k':R,'nonpositive':tot-pos})
    ks=min(k for k in range(2,13) if all(all_R[j]==1 for j in range(k,13)))
    kstars[(side,route)]=ks
pd.DataFrame(sparse).to_csv(DATA/'Qatraneh_R15R4_SparseSignRetention.csv',index=False)

# Legacy R15R2 decision record retained for fixed-table figure reconstruction only;
# separately reproduced 70/70 by recompute_qatraneh_gep.py.
windows=[6,7,8,9,10,12,14]
model_rows=[
('North','Field K',['E','E','G','G','G','P','P'],'P',12),
('North','Fine LF',['G','G','P','P','G','P','P'],'P',12),
('North','Fine HF',['G','G','P','P','G','P','P'],'P',12),
('North','Coarse LF',['G','E','P','P','G','G','G'],'G',10),
('North','Coarse HF',['G','E','P','P','G','G','G'],'G',10),
('South','Field K',['G','G','G','P','P','P','P'],'P',9),
('South','Fine LF',['G','P','P','P','P','P','P'],'P',7),
('South','Fine HF',['G','P','P','P','P','P','P'],'P',7),
('South','Coarse LF',['G','G','P','P','P','P','P'],'P',8),
('South','Coarse HF',['G','G','G','P','P','P','P'],'P',9),
]
mr=[]
for side,route,seq,full,dstar in model_rows:
    for w,m in zip(windows,seq): mr.append({'Side':side,'Route':route,'Outer_window_m':w,'Model':m,'Full_model':full,'Dstar_m':dstar})
pd.DataFrame(mr).to_csv(DATA/'Qatraneh_R15R4_NestedModelDecisions_LOCKED.csv',index=False)

integ=[]
sumdf=pd.DataFrame(summary)
for side,route,seq,full,dstar in model_rows:
    rec=sumdf[(sumdf.Side==side)&(sumdf.Route==route)].iloc[0]
    integ.append({'Side':side,'Route':route,'Full_model':full,'kstar':kstars[(side,route)],'Dstar_m':dstar,'C10':rec.C10,'Outer_max':rec.Outer_max})
integ=pd.DataFrame(integ)
integ.to_csv(DATA/'Qatraneh_R15R4_IntegratedResults.csv',index=False)

# Daejeon fixed summary: output preservation only, not workbook-to-result replay
D=pd.DataFrame([
['S1','-', 'G',35.6,33.7,16,34],
['S2','-', 'E',6.0,1.0,4,25],
['N1','+', 'E',57.1,97.2,10,35],
['N2','-', 'E',37.5,2.0,0,48],
['N3','-', 'E',55.4,91.4,1,34]], columns=['Profile','Sign','Full_model','Model_stable_m','Sign_stable_m','Tail_best','Eligible'])
D.to_csv(DATA/'Daejeon_R15R4_LockedSummary.csv',index=False)

# Daejeon fixed reconciliation record: no source-workbook extraction is performed here
R=pd.DataFrame([
['A','S1',40,'+5','Mean 29.22; median 28.6; SD 13.02; maximum 71.7 at 11.6 m','Admit: count and published mean/median/dispersion agree closely'],
['B','S2',31,'0','Mean 25.37; median 21.4; SD 13.54','Admit with dispersion flag: count, mean and median agree; published SD is 8.6'],
['C','N1',41,'+5','Mean 29.91; median 29.0; SD 10.16','Admit with count flag: mean/median agree; article reports 40 readings'],
['D','S3 candidate',32,'+5','Mean 27.53; median 26.55; 32 values duplicate a segment of run C','Exclude: fails published S3 summary and contains duplicated N1 susceptibility sequence'],
['E','N2',54,'0','Mean 27.10; median 25.2; SD 10.13','Admit: count, mean and median match published profile'],
['F','N3',40,'-5','Mean 30.18; median 29.25; SD 8.26','Admit with count/median flag: article reports 41 readings and median 28.3']],
columns=['Run','Reconciled_profile','Rows','Separation_m','Workbook_check','Decision'])
R.to_csv(DATA/'Daejeon_R15R4_SourceReconciliation.csv',index=False)

# Figure 1: observed profiles (stacked panels for readability)
fig,axs=plt.subplots(2,1,figsize=(9.4,9.2))
for ax, profile, ylabel, title in [
    (axs[0],'Field K',r'$K_{\mathrm{field}}$ ($\times 10^{-6}$ SI)','(a) Field volume susceptibility'),
    (axs[1],'Fine LF',r'Fine $\chi_{\mathrm{LF}}$ ($\times 10^{-8}$ m$^3$ kg$^{-1}$)','(b) Fine-fraction mass susceptibility')]:
    ax.plot(dist,north[profile],marker='o',markersize=7.5,linewidth=2.2,label='North')
    ax.plot(dist,south[profile],marker='s',markersize=7.5,linewidth=2.2,label='South')
    ax.set_xlabel('Distance from pavement edge (m)'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=.22,linewidth=0.9); ax.legend(frameon=False); style_axis(ax)
fig.tight_layout(h_pad=2.0); fig.savefig(FIG/'Figure1_Qatraneh_Bilateral_Profiles.pdf',bbox_inches='tight'); fig.savefig(FIG/'Figure1_Qatraneh_Bilateral_Profiles.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure 2: observed-maximum comparison and model decisions
profile_labels=[f"{s[0]} - {r}" for s,r,_,_,_ in model_rows]
mat=np.array([[{'G':0,'P':1,'E':2}[m] for m in seq] for _,_,seq,_,_ in model_rows])
fig,axs=plt.subplots(2,1,figsize=(10.8,11.2),gridspec_kw={'height_ratios':[1.35,1]})
ax=axs[0]
im=ax.imshow(mat,aspect='auto',interpolation='nearest',cmap='viridis',vmin=0,vmax=2)
ax.set_xticks(range(len(windows)),windows,fontsize=12,fontweight='bold'); ax.set_yticks(range(len(profile_labels)),profile_labels,fontsize=11,fontweight='bold')
ax.set_xlabel('Outer sampled distance (m)'); ax.set_title('(a) Model decision as distal stations are revealed')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j,i,['G','P','E'][mat[i,j]],ha='center',va='center',fontweight='bold',fontsize=12,
                color='white' if mat[i,j] in (0,1) else 'black')
ax.text(0.0,-0.12,'G = gradient   P = ramp-to-plateau   E = edge-decay',transform=ax.transAxes,fontsize=12,fontweight='bold')
style_axis(ax)
ax=axs[1]
offsets={
('North','Field K'):(6,10),('North','Fine LF'):(6,-1),('North','Fine HF'):(6,-12),
('North','Coarse LF'):(6,9),('North','Coarse HF'):(6,-11),
('South','Field K'):(6,9),('South','Fine LF'):(6,-12),('South','Fine HF'):(6,-12),
('South','Coarse LF'):(6,9),('South','Coarse HF'):(6,-11),
}
for i,row in integ.iterrows():
    stable=row['Dstar_m']<=10
    marker='o' if stable else 'X'
    ax.scatter(row['C10']*100,row['Dstar_m'],s=85,marker=marker)
    label=('N' if row.Side=='North' else 'S')+'-'+row.Route.replace('Field K','Field').replace(' ','')
    ax.annotate(label,(row['C10']*100,row['Dstar_m']),xytext=offsets[(row.Side,row.Route)],textcoords='offset points',fontsize=10.5,fontweight='bold')
ax.axvline(90,linestyle='--',linewidth=1); ax.axhline(10,linestyle='--',linewidth=1)
ax.set_xlim(89,103.5); ax.set_ylim(6,13); ax.set_xlabel('Complete-record maximum encountered by 10 m (%)'); ax.set_ylabel(r'Model stabilization $D^*$ (m)')
ax.set_title('(b) Observed maximum and model stability differ')
ax.text(89.3,12.6,'10/10 profiles >=90% by 10 m\n7/10 model-stable by 10 m',fontsize=12,fontweight='bold',va='top')
style_axis(ax)
fig.tight_layout(h_pad=2.2); fig.savefig(FIG/'Figure2_ObservedMaximum_and_ModelStability.pdf',bbox_inches='tight'); fig.savefig(FIG/'Figure2_ObservedMaximum_and_ModelStability.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/'Figure2_ObservedMaximum_and_ModelStability.svg',bbox_inches='tight'); plt.close(fig)

# Figure 3: station placement result, stacked panels for readability
fig,axs=plt.subplots(2,1,figsize=(9.6,9.4))
field=sparse_df=pd.DataFrame(sparse)
neg=[]
for k in [2,3,4,5]:
    q=field[(field.Route=='Field K')&(field.k==k)]
    neg.append(int(q.nonpositive.sum()))
axs[0].bar([2,3,4,5],neg)
axs[0].set_xticks([2,3,4,5]); axs[0].set_xlabel('Stations retained, k'); axs[0].set_ylabel('Negative field-profile layouts\n(north + south)'); axs[0].set_ylim(0,16.5)
axs[0].set_title('(a) Sparse layouts can reverse the observed direction')
for x,y in zip([2,3,4,5],neg): axs[0].text(x,y+0.4,str(y),ha='center',fontsize=11,fontweight='bold')
style_axis(axs[0])
labels=[('N' if s=='North' else 'S')+'-'+r.replace('Field K','Field').replace(' ','') for s,r,_,_,_ in model_rows]
kvals=[kstars[(s,r)] for s,r,_,_,_ in model_rows]
axs[1].barh(range(len(labels)),kvals)
axs[1].set_yticks(range(len(labels)),labels); axs[1].invert_yaxis(); axs[1].set_xlabel(r'Strict sign-invariance $k^*$ (stations)')
axs[1].set_title('(b) Direction recoverability is profile-specific')
axs[1].set_xlim(0,12)
for i,v in enumerate(kvals): axs[1].text(v+0.15,i,str(v),va='center',fontsize=10.5,fontweight='bold')
style_axis(axs[1])
fig.tight_layout(h_pad=2.0); fig.savefig(FIG/'Figure3_StationPlacement_DirectionRecoverability.pdf',bbox_inches='tight'); fig.savefig(FIG/'Figure3_StationPlacement_DirectionRecoverability.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure 4: Daejeon (single panel already clear)
fig,ax=plt.subplots(figsize=(9.4,5.6))
y=np.arange(len(D))
ax.hlines(y,D.Model_stable_m,D.Sign_stable_m,linewidth=1.8,alpha=.65)
ax.scatter(D.Model_stable_m,y,marker='o',s=90,label='Model decision')
ax.scatter(D.Sign_stable_m,y,marker='x',s=100,linewidths=2.2,label='Direction sign')
ax.set_yticks(y,D.Profile); ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel('Stabilization distance (m)')
ax.set_title('Illustrative Daejeon profiles: different inferences stabilize at different distances')
ax.legend(frameon=False); ax.grid(axis='x',alpha=.22,linewidth=0.9); style_axis(ax)
fig.tight_layout(); fig.savefig(FIG/'Figure4_Daejeon_Stabilization.pdf',bbox_inches='tight'); fig.savefig(FIG/'Figure4_Daejeon_Stabilization.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Verification summary
checks={
'rows':int(len(obs)), 'complete_positive_routes':int(sum(ts(x,y)>0 for _,_,x,y in routes)),
'cutpoint_positive':int(sum(r['Outer_minus_inner']>0 for r in cut)),
'cutpoint_total':int(len(cut)), 'loo_positive':int(sum(r['TheilSen_slope']>0 for r in loo)), 'loo_total':int(len(loo)),
'sparse_fit_total':int(sum(r['total'] for r in sparse)),
'kstar_min':int(min(kstars.values())),'kstar_max':int(max(kstars.values())),
'Observed_max_fraction_ge_0.9_by_10m':int((integ.C10>=.9).sum()),'Dstar_le_10':int((integ.Dstar_m<=10).sum()),
'outer_max_count':int(integ.Outer_max.sum()),
}
(DATA/'Qatraneh_R15R4_Verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
