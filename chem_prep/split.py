"""Group-preserving stratified splits with explicit infeasibility reporting."""
import hashlib,json
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix,hstack,vstack,eye
from .backend import create_backend
from .chemistry import load_molecule
from .io import write_json,write_manifest


def identity(smiles):
    ig,_=create_backend();mol,_=load_molecule(ig,smiles);aromatic=mol.clone()
    try:aromatic.aromatize();mol=aromatic
    except Exception:pass
    try:value='canonical:'+mol.canonicalSmiles()
    except Exception:
        import networkx as nx
        g=nx.Graph()
        for a in mol.iterateAtoms():g.add_node(a.index(),label=json.dumps([a.symbol(),a.charge(),a.isotope(),a.isPseudoatom()]))
        for b in mol.iterateBonds():g.add_edge(b.source().index(),b.destination().index(),label=str(b.bondOrder()))
        value='conservative-WL:'+nx.weisfeiler_lehman_graph_hash(g,node_attr='label',edge_attr='label',iterations=6)
    return hashlib.sha256(value.encode()).hexdigest()


class UnionFind:
    def __init__(self,n):self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)


def assign(root,rows,settings):
    uf=UnionFind(len(rows));seen={};parents={r['sample_id']:i for i,r in enumerate(rows)};cache={}
    for i,r in enumerate(rows):
        if r.get('parent_id') in parents:uf.union(i,parents[r['parent_id']])
        for smiles in {r['source_SMILES'],r['SMILES']}:
            if smiles not in cache:cache[smiles]=identity(smiles)
            key=cache[smiles]
            if key in seen:uf.union(i,seen[key])
            else:seen[key]=i
    members=defaultdict(list)
    for i in range(len(rows)):members[uf.find(i)].append(i)
    groups=sorted(members.values(),key=lambda g:min(rows[i]['sample_id'] for i in g));ng=len(groups);fraction=settings['validation_fraction']
    fields=settings['stratify'];allowed={'dataset','variant','is_color','dummy_applied','has_charge'}
    if not set(fields)<=allowed:raise ValueError(f'Unsupported stratification fields: {set(fields)-allowed}')
    strata=sorted({tuple(str(r.get(f,'')) for f in fields) for r in rows});lookup={s:i for i,s in enumerate(strata)};counts=Counter(tuple(str(r.get(f,'')) for f in fields) for r in rows)
    floors={s:int(counts[s]*fraction) for s in strata};target=round(len(rows)*fraction)
    for s in sorted(strata,key=lambda s:(-(counts[s]*fraction-floors[s]),s))[:target-sum(floors.values())]:floors[s]+=1
    rr=[];cc=[];vv=[]
    for gi,g in enumerate(groups):
        weights=Counter(lookup[tuple(str(rows[i].get(f,'')) for f in fields)] for i in g)
        for k,n in weights.items():rr.append(k);cc.append(gi);vv.append(n)
    a=coo_matrix((np.array(vv,float),(rr,cc)),shape=(len(strata),ng)).tocsc();targets=np.array([floors[s] for s in strata],float)
    rng=np.random.default_rng(settings['seed']);cost=(1-2*(rng.random(ng)<fraction).astype(float))+rng.random(ng)*1e-6
    exact=False
    if ng==1:chosen=np.zeros(1,int);message='Only one molecule group: retained in train; nonempty validation impossible'
    else:
        nonempty=LinearConstraint(np.ones((1,ng)),[1],[ng-1])
        result=milp(cost,integrality=np.ones(ng),bounds=Bounds(np.zeros(ng),np.ones(ng)),constraints=[LinearConstraint(a,targets,targets),nonempty],options={'time_limit':settings['solver_seconds']})
        if result.x is not None and np.max(np.abs(a@np.rint(result.x)-targets))<1e-6:chosen=np.rint(result.x).astype(int);exact=True;message=result.message
        else:
            # Never split a molecule group to force a rounded count. Minimize
            # per-stratum count error explicitly and report the achieved ratio.
            ns=len(strata);matrix=hstack([a,-eye(ns),eye(ns)]).tocsc();objective=np.r_[cost*1e-4,np.ones(ns),np.ones(ns)]
            ones=coo_matrix(np.r_[np.ones(ng),np.zeros(2*ns)].reshape(1,-1)).tocsc()
            fallback_constraints=[LinearConstraint(matrix,targets,targets),LinearConstraint(ones,[1],[ng-1])]
            # Preserve the requested global count even when rounded strata are
            # indivisible. Relax it only if whole-group sizes make it impossible.
            total_row=coo_matrix(np.r_[np.asarray(a.sum(axis=0)).ravel(),np.zeros(2*ns)].reshape(1,-1)).tocsc()
            result=milp(objective,integrality=np.r_[np.ones(ng),np.zeros(2*ns)],bounds=Bounds(np.zeros(ng+2*ns),np.r_[np.ones(ng),np.full(2*ns,np.inf)]),constraints=fallback_constraints+[LinearConstraint(total_row,[target],[target])],options={'time_limit':settings['solver_seconds']})
            if result.x is None:
                result=milp(objective,integrality=np.r_[np.ones(ng),np.zeros(2*ns)],bounds=Bounds(np.zeros(ng+2*ns),np.r_[np.ones(ng),np.full(2*ns,np.inf)]),constraints=fallback_constraints,options={'time_limit':settings['solver_seconds']})
            if result.x is None:raise RuntimeError('Could not find group-preserving split: '+result.message)
            chosen=np.rint(result.x[:ng]).astype(int);message='Rounded stratum targets infeasible; used group-preserving minimum-deviation split'
    for gi,g in enumerate(groups):
        keys=sorted({cache[rows[i]['source_SMILES']] for i in g}|{cache[rows[i]['SMILES']] for i in g});gid=hashlib.sha256('\n'.join(keys).encode()).hexdigest()
        for i in g:rows[i]['group']=gid;rows[i]['split']='val' if chosen[gi] else 'train'
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    for smileskey in ['source_SMILES','SMILES']:
        if {cache[r[smileskey]] for r in train}&{cache[r[smileskey]] for r in val}:raise RuntimeError('Split leakage')
    report={'groups':ng,'largest_group':max(map(len,groups)),'requested_validation_fraction':fraction,'actual_validation_fraction':len(val)/len(rows),'exact_stratum_counts':exact,'message':message,'train':len(train),'val':len(val),'strata':[{'key':s,'requested_val':floors[s],'actual_val':sum(tuple(str(r.get(f,'')) for f in fields)==s for r in val)} for s in strata]}
    write_json(root/'molecule_identities.json',cache);write_manifest(root,'train',train);write_manifest(root,'val',val);write_json(root/'split_report.json',report)
    return train,val,report
