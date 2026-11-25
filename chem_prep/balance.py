"""Explicit-epoch balancing with fixed validation, preserved color/dummy mix."""
import copy,math,random
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from .io import file_hashes,read_json,write_json,write_manifest


def snapshot_validation(root):
    root=Path(root);val=read_json(root/'val.json');path=root/'holdout_snapshot.json'
    hashes=file_hashes(root,val,['val.json','val.csv'])
    if path.exists():
        prior=read_json(path)
        if hashes!=prior['files']:raise RuntimeError('Frozen validation assets or manifests changed')
        return prior
    data={'rows':len(val),'files':hashes};write_json(path,data);return data


def verify_holdout(root):
    before=read_json(Path(root)/'holdout_snapshot.json');after=file_hashes(root,read_json(Path(root)/'val.json'),['val.json','val.csv'])
    changed=[p for p,h in before['files'].items() if after.get(p)!=h]
    if changed or set(after)!=set(before['files']):raise RuntimeError(f'Validation changed: {changed[:5]}')
    return {'status':'byte_identical','rows':before['rows'],'files_checked':len(after),'changed_files':[]}


def stratum(row):return 'dummy_color' if row['is_color'] and row['dummy_applied'] else 'plain_color' if row['is_color'] else 'monochrome'


def make_epoch(original,unique,target,seed):
    if not 0<target<1:raise ValueError('charged_fraction must be in (0,1)')
    cells=[(positive,s) for positive in [False,True] for s in ['monochrome','plain_color','dummy_color']]
    pools={cell:[r for r in unique if (r['has_charge'],stratum(r))==cell] for cell in cells}
    pos=sum(r['has_charge'] for r in unique);neg=len(unique)-pos
    if not pos or not neg:raise ValueError('Charge balancing requires both charged and uncharged training images; cannot create missing chemistry classes by sampling')
    frac=Fraction(str(target));k,den=frac.numerator,frac.denominator
    if den>10000:raise ValueError('Use a charged_fraction with at most four decimal places')
    units=max(math.ceil(pos/k),math.ceil(neg/(den-k)))
    color_fraction=sum(r['is_color'] for r in original)/len(original);dummy_fraction=sum(r['is_color'] and r['dummy_applied'] for r in original)/len(original)
    minimum=np.array([len(pools[c]) for c in cells],float);upper=np.array([np.inf if pools[c] else 0 for c in cells])
    a=np.array([[int(c[0]) for c in cells],[int(not c[0]) for c in cells],[int(c[1]!='monochrome') for c in cells],[int(c[1]=='dummy_color') for c in cells]],float)
    chosen=None;relaxed=False
    proportions=np.array([target,1-target,color_fraction,dummy_fraction])
    feasibility=milp(np.zeros(6),integrality=np.zeros(6),bounds=Bounds(np.zeros(6),np.where(upper==0,0,np.inf)),constraints=LinearConstraint(a,proportions,proportions),options={'time_limit':2})
    for attempt in range(12 if feasibility.x is not None else 0):
        total=units*den;np_=units*k;nn=total-np_;targets=np.array([np_,nn,round(total*color_fraction),round(total*dummy_fraction)],float)
        original_counts=Counter((r['has_charge'],stratum(r)) for r in original)
        preferred=np.array([(np_ if c[0] else nn)*original_counts[c]/max(1,sum(original_counts[x] for x in cells if x[0]==c[0])) for c in cells])
        # q - d+ + d- = preferred: choose integer strata close to the
        # original within-class distribution, while honoring class/global mix.
        matrix=np.vstack([np.hstack([a,np.zeros((4,12))]),np.hstack([np.eye(6),-np.eye(6),np.eye(6)])])
        result=milp(np.r_[np.zeros(6),np.ones(12)],integrality=np.r_[np.ones(6),np.zeros(12)],bounds=Bounds(np.r_[minimum,np.zeros(12)],np.r_[upper,np.full(12,np.inf)]),constraints=LinearConstraint(matrix,np.r_[targets,preferred],np.r_[targets,preferred]),options={'time_limit':5})
        if result.x is not None:chosen=np.rint(result.x[:6]).astype(int);break
        units=max(units+1,math.ceil(units*1.03))
    if chosen is None:
        # Missing class/color combinations can make the original global mix
        # incompatible with the new charged fraction. Keep the class fraction
        # exact, retain all unique assets, and report the closest color mix.
        total=units*den;np_=units*k;nn=total-np_
        targets=np.array([np_,nn,round(total*color_fraction),round(total*dummy_fraction)],float)
        matrix=np.vstack([np.hstack([a[:2],np.zeros((2,4))]),np.hstack([a[2:],-np.eye(2),np.eye(2)])])
        res=milp(np.r_[np.zeros(6),np.ones(4)],integrality=np.r_[np.ones(6),np.zeros(4)],bounds=Bounds(np.r_[minimum,np.zeros(4)],np.r_[upper,np.full(4,np.inf)]),constraints=LinearConstraint(matrix,targets,targets),options={'time_limit':10})
        if res.x is None:raise RuntimeError('Cannot satisfy charged fraction with available sampling strata')
        chosen=np.rint(res.x[:6]).astype(int);relaxed=True
    rng=random.Random(seed);draws=[]
    for cell,n in zip(cells,chosen):
        pool=pools[cell];part=list(pool)
        while len(part)<n:
            cycle=list(pool);rng.shuffle(cycle);part.extend(cycle[:n-len(part)])
        draws.extend(part)
    counter=Counter();epoch=[]
    for row in draws:
        r=copy.deepcopy(row);asset=r['sample_id'];idx=counter[asset];counter[asset]+=1
        r.update({'asset_sample_id':asset,'sampling_repeat_index':idx,'sample_id':asset if not idx else asset+f'::repeat_{idx:04d}'})
        epoch.append(r)
    rng.shuffle(epoch)
    if Fraction(sum(r['has_charge'] for r in epoch),len(epoch))!=frac:raise RuntimeError('Charge sampling target not met')
    return epoch,{'target_charged_fraction':target,'actual_charged_fraction':sum(r['has_charge'] for r in epoch)/len(epoch),'sampling_rows':len(epoch),'unique_training_images':len(unique),'charged_rows':sum(r['has_charge'] for r in epoch),'repeated_rows':len(epoch)-len(unique),'color_fraction':sum(r['is_color'] for r in epoch)/len(epoch),'dummy_fraction':sum(r['dummy_applied'] for r in epoch)/len(epoch),'quotas':[{'has_charge':c[0],'stratum':c[1],'rows':int(n)} for c,n in zip(cells,chosen)],'color_mix_relaxed':relaxed,'original_color_fraction':color_fraction,'original_dummy_fraction':dummy_fraction,'method':'explicit_epoch_manifest; ordinary uniform DataLoader with shuffle; no extra weighted sampler'}


def apply(root,unique,settings):
    root=Path(root);snapshot_validation(root);original=read_json(root/'source_train.json');val=read_json(root/'val.json')
    if {r['group'] for r in unique}&{r['group'] for r in val}:raise RuntimeError('Training augmentation leaked into validation groups')
    epoch,report=make_epoch(original,unique,settings['charged_fraction'],settings['seed'])
    write_manifest(root,'train',epoch);write_json(root/'unique_train.json',unique)
    report['validation']=verify_holdout(root);write_json(root/'balance_report.json',report)
    return report
