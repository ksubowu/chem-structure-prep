"""Resumable bounded native styles + one aligned affine view per real CID style."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from runtime import ROOT,BASE,AUDIT,SOURCE,DATA,TOOL,workers
import sys,json,hashlib,random,collections,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(TOOL))
from chem_prep.render import render_one
from chem_prep.charges import generate as typography
from chem_prep.lines import extend_paths,write_artifacts
from chem_prep.io import write_json,seed_for
from chem_prep.visuals import overlay,review
from PIL import Image
import numpy as np
from chem_prep.backend import create_backend
from geometric_views import generate as geometric

IG=None;HELD=None
FIELDS=['image','annotation','svg','masks','band_target','line_target','overlay','line_overlay','view_check','band_line_view_check']
LABELS=['CF3','C2H5','OMe','OtBu','Ph','Me','Et','iPr','tBu','CO2Me','COOH','NHAc','Ac','Boc','Cbz','TMS','Ts','SO2Me','SO2NH2']
def init():
 global IG,HELD
 IG,_=create_backend()
 if (ROOT/'excluded_native_identities.json').exists():HELD=set(json.loads((ROOT/'excluded_native_identities.json').read_text()))
 else:
  cache=json.loads((AUDIT/'validation_reference/molecule_identities.json').read_text());HELD={cache[x[k]] for x in json.loads((AUDIT/'validation_reference/val.json').read_text()) for k in ['source_SMILES','SMILES']}
def norm(m):
 c=m.clone()
 try:c.aromatize()
 except Exception:pass
 return hashlib.sha256(('canonical:'+c.canonicalSmiles()).encode()).hexdigest()
def counts(a):return {f:dict(c) for f,c in [('atoms',collections.Counter(x['symbol'] for x in a['atoms'] if not x.get('is_pseudoatom'))),('charges',collections.Counter(str(q['value']) for q in a['charges'])),('bonds',collections.Counter(str(b['category_id']) for b in a['bands']))]}
def clear(row):
 for field in FIELDS:
  if field in row and row[field].startswith(('assets/pubchem/','assets/class_aug/','assets/charge/')):
   # Only newly produced PubChem files are passed here, never inherited rows.
   (DATA/row[field]).unlink(missing_ok=True)
def pad_charge(row):
 path=DATA/row['annotation'];a=json.loads(path.read_text());changed=False
 for q in a['charges']:
  x1,y1,x2,y2=q['bbox'];delta=max(0.,6.-(y2-y1))/2
  if delta:q['bbox']=[x1,max(0.,y1-delta),x2,min(float(a['height']),y2+delta)];changed=True
 if changed:
  write_json(path,a)
  rgb=np.array(Image.open(DATA/row['image']).convert('RGB'))
  with np.load(DATA/row['masks']) as mm:marked=overlay(rgb,a,mm)
  marked.save(DATA/row['overlay']);review(None,rgb,marked,row['sample_id']).save(DATA/row['view_check'],quality=90);write_artifacts(DATA,a,row)
 return a

def worker(job):
 p=job['parent'];slot=job['slot'];sid=f"pubchem/{p['pubchem_cid']}/style{slot}";key=hashlib.sha256(sid.encode()).hexdigest();checkpoint=DATA/'_pubchem_checkpoints'/(key+'.json')
 if checkpoint.exists():
  old=json.loads(checkpoint.read_text())
  if old.get('status')=='ok' and all((DATA/r[k]).is_file() for r in old['rows'] for k in FIELDS):return old
 rng=random.Random(job['seed']);stem=f"cid{p['pubchem_cid']}_style{slot}";paths={k:f'assets/pubchem/{folder}/{stem}.{ext}' for k,folder,ext in [('image','images','png'),('annotation','annotations','json'),('masks','masks','npz'),('svg','svg','svg'),('band_target','targets','json'),('overlay','overlays','png'),('view_check','view_check','jpg')]};paths.update(extend_paths(paths));selected=job['notation'];errors=[];native=None
 # Different valid equivalent source strings can avoid a native stereo/layout
 # anomaly. Alias-free color is the last fallback; do not relabel failed bonds.
 tries=[(selected['SMILES'],job['is_color'],.5)]+[(p['SMILES'],job['is_color'],0),(selected['SMILES'],False,0)]
 for attempt,(sm,color,dummyprob) in enumerate(tries):
  setting={'canvas_size':640,'padding':rng.randint(26,72),'font_families':[['DejaVu Sans','DejaVu Serif','DejaVu Sans Mono'][(job['seed']+slot)%3]],'relative_thickness':rng.uniform(.80,1.45),'bond_length':rng.uniform(36,54),'match_reference':False,'max_atoms':140,'reject_non_single_wedges':True}
  request={'root':str(DATA),'paths':paths,'render':setting,'source':{'dataset':'PubChem','id':stem,'smiles':sm},'seed':job['seed'],'is_color':color,'dummy_probability':dummyprob,'labels':LABELS,'sample_id':sid,'variant':'pubchem_native','fingerprint':sid}
  r=render_one(request)
  if r['status']!='ok':errors.append(r.get('error'));continue
  try:
   a=json.loads((DATA/r['annotation']).read_text());m=IG.loadMolecule(a['rendered_molfile']);aug=a.get('augmentation')
   actual=norm(IG.loadMolecule(a['SMILES']))
   if actual in HELD:raise ValueError('Rendered chemistry overlaps frozen validation')
   if aug and aug.get('applied'):
    d=m.getAtom(aug['dummy_atom_id'])
    if not d.isPseudoatom():raise ValueError('Functional label is not a dummy')
    d.resetAtom('H');m.foldHydrogens();restored=norm(m)
   else:restored=actual
   if restored!=p['native_identity']:raise ValueError('Source chemistry/stereochemistry mismatch')
   r.update(group=p['native_identity'],original_parent_id=p['native_identity'],parent_id=p['native_identity'],pubchem_cid=p['pubchem_cid'],raw_id=p['raw_id'],scaffold_cluster=p['scaffold_cluster'],source_pubchem_SMILES=p['SMILES'],source_SMILES=p['SMILES'],split='train',class_balance_mode=selected['mode'],chemical_equivalence_verified=True,planned_views=1)
   a['source_SMILES']=p['SMILES'];a['pubchem_provenance']={'CID':p['pubchem_cid'],'raw_id':p['raw_id'],'source_SMILES':p['SMILES'],'parent_native_identity':p['native_identity'],'scaffold_cluster':p['scaffold_cluster'],'notation_mode':selected['mode'],'alias_reversal_identity_verified':True,'style_slot':slot};write_json(DATA/r['annotation'],a);native=r;break
  except Exception as ex:errors.append(str(ex));clear(r)
 if native is None:
  result={'status':'error','pubchem_cid':p['pubchem_cid'],'slot':slot,'errors':errors};write_json(checkpoint,result);return result
 typo_error=None
 if native['has_charge']:
  ti=seed_for(job['seed'],'charge');tr=typography((native,str(DATA),str(DATA),int(p['pubchem_cid']),slot,ti,{'canvas_size':640,'size_tier':'large' if rng.random()<.35 else 'medium'}))
  if tr['status']=='ok':
   clear(native);native=tr;native['variant']='pubchem_charge_style'
  else:typo_error=tr.get('error')
 a=pad_charge(native);out=[native];detail=[counts(a)];g=geometric((native,str(DATA),0))
 if g['status']=='ok':
  ga=pad_charge(g)
  if any(b['order']!=1 and b['stereo'] in (5,6) for b in ga['bonds']):raise ValueError('Unexpected non-single wedge after affine transform')
  g['variant']='pubchem_affine_style';out.append(g);detail.append(counts(ga))
 result={'status':'ok','rows':out,'subclass_counts':detail,'fallback_errors':errors,'charge_typography_fallback':typo_error,'geometric_fallback':None if g['status']=='ok' else g,'pubchem_cid':p['pubchem_cid'],'slot':slot};write_json(checkpoint,result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args();jobs=json.loads((ROOT/'selected_jobs.json').read_text())
 if args.limit:
  # Diverse benchmark: take evenly spaced jobs across all source strata.
  import numpy as np
  jobs=[jobs[i] for i in np.linspace(0,len(jobs)-1,min(args.limit,len(jobs)),dtype=int)]
 start=time.monotonic();print('Native styles',len(jobs),'workers',workers(192),flush=True);results=[]
 with ProcessPoolExecutor(max_workers=workers(192),initializer=init) as pool:
  for i,r in enumerate(pool.map(worker,jobs,chunksize=2),1):
   results.append(r)
   if i%250==0:print('Completed',i,'/',len(jobs),'seconds',round(time.monotonic()-start,1),flush=True)
 stem='benchmark' if args.limit else 'generation';write_json(ROOT/(stem+'_results.json'),results);bad=[r for r in results if r['status']!='ok'];write_json(ROOT/(stem+'_errors.json'),bad);rows=[x for r in results if r['status']=='ok' for x in r['rows']];write_json(ROOT/(stem+'_manifest.json'),rows);print(json.dumps({'native_styles':len(jobs),'successful_images':len(rows),'failed_styles':len(bad),'seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
