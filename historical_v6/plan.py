"""Diversity-first long-tail enrichment; no equal-instance ratio target."""
from pathlib import Path
import json,collections,hashlib,random
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;AUDIT=BASE/'pubchem_diversity_audit_20261121';SOURCE=BASE/'indigo_subclass_balance_v5_20261120/dataset'
ATOMS=['C','O','N','Cl','Br','S','F','B','I','P','H','Si'];KEYS=[('atoms',k) for k in ATOMS]+[('charges',k) for k in ['-1','1']]+[('bonds',str(k)) for k in range(1,7)]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 native=json.loads((AUDIT/'native_probe_results.json').read_text())+json.loads((ROOT/'native_extra_results.json').read_text());rows=[];jobs=[]
 for r in native:
  valid=[v for v in r.get('variants',[]) if 'error' not in v and set(v['charges'])<=set(['-1','1'])]
  if not valid:continue
  rng=random.Random(int(r['native_identity'][:16],16));normal=[v for v in valid if v['mode']!='aromatic_explicit_H'];normal=normal or valid;exp=[v for v in valid if v['mode']=='aromatic_explicit_H'];v=rng.choice(exp if exp and rng.random()<.40 else normal)
  extra=any(v['atoms'].get(k,0)>0 for k in ['B','Br']) or v['bonds'].get('3',0)>0 or bool(v['charges'])
  slots=4 if v['atoms'].get('B') else 2 if extra else 1
  for slot in range(slots):
   chosen=v if not slot else rng.choice([x for x in valid if x['mode']!=v['mode']] or valid);color=rng.random()<.30

   jobs.append({'parent':{k:r[k] for k in ['raw_id','pubchem_cid','SMILES','native_identity','scaffold_cluster','heavy','rings']},'notation':chosen,'slot':slot,'is_color':color,'seed':int(hashlib.sha256((r['native_identity']+'/'+str(slot)).encode()).hexdigest()[:15],16),'planned_views':1})
  rows.append({'native_identity':r['native_identity'],'pubchem_cid':r['pubchem_cid'],'scaffold_cluster':r['scaffold_cluster'],'native_jobs':slots,'final_images':slots*2,'charged':bool(v['charges'])})
 # Raise charged-image coverage through different real charged parents, not repeats.
 base=json.loads((SOURCE/'source_train.json').read_text());target=.22;existing=collections.Counter(j['parent']['native_identity'] for j in jobs);by_identity={r['native_identity']:r for r in native if 'native_identity' in r};current=sum(x['has_charge'] for x in base)+2*sum(bool(j['notation']['charges']) for j in jobs)
 candidates=sorted([r for r in rows if r['charged'] and r['native_jobs']<3],key=lambda r:r['native_identity'])
 for row in candidates:
  if current/(len(base)+2*len(jobs))>=target:break
  r=by_identity[row['native_identity']];vs=[v for v in r['variants'] if 'error' not in v and set(v['charges'])<=set(['-1','1'])];seed=int(r['native_identity'][:15],16)+20261121;rr=random.Random(seed);v=rr.choice(vs);slot=row['native_jobs'];jobs.append({'parent':{k:r[k] for k in ['raw_id','pubchem_cid','SMILES','native_identity','scaffold_cluster','heavy','rings']},'notation':v,'slot':slot,'is_color':rr.random()<.30,'seed':seed,'planned_views':1});row['native_jobs']+=1;row['final_images']+=2;current+=2
 old=json.loads((AUDIT/'chemical_statistics.json').read_text());profile=json.loads((BASE/'indigo_balance_v4_20261120/profile_before.json').read_text());counts={f:collections.Counter(c) for f,c in profile['distributions']['train_unique']['instance_counts'].items()}
 for j in jobs:
  for fam,key in KEYS:counts[fam][key]+=2*j['notation'][fam].get(key,0)
 totals={f:sum(c[k] for ff,k in KEYS if ff==f) for f,c in counts.items()};fractions={f:{k:counts[f][k]/totals[f] for ff,k in KEYS if ff==f} for f in counts};base=json.loads((SOURCE/'source_train.json').read_text());total=len(base)+2*len(jobs);charged=sum(r['has_charge'] for r in base)+sum(2*bool(j['notation']['charges']) for j in jobs)
 report={'policy':'Retain natural dominance of C/single; enrich rare real molecular sources, limited 2–8 unique images per PubChem chemical; no repeated epoch entries.','original_unique_training_images':len(base),'new_real_chemical_identities':len(rows),'new_scaffold_clusters':len({r['scaffold_cluster'] for r in rows}),'native_render_jobs':len(jobs),'projected_unique_training_images':total,'projected_charged_fraction':charged/total,'projected_counts_before_aliases':{f:dict(c) for f,c in counts.items()},'projected_subclass_fractions':fractions,'max_images_per_new_chemical':8,'min_images_per_new_chemical':2,'validation_reference':'Frozen v5 1498 rows; conservative union with current v3 1497; no changes.','previous_29_template_expansion_included':False,'missing_charge_classes_report_only':[-4,-2,2]}
 write(ROOT/'selected_jobs.json',jobs);write(ROOT/'selected_molecules.json',rows);write(ROOT/'plan_report.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
