"""Complete per-image artifact audit, chemical split audit and count cross-check."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from runtime import ROOT,DATA,TOOL,workers
import sys,json,collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(TOOL))
from chem_prep.split import identity
from chem_prep.validate import validate
from chem_prep.io import write_json
from plan import KEYS
if __name__=='__main__':
 rows=json.loads((DATA/'data_manifest.json').read_text());cache=json.loads((DATA/'molecule_identities.json').read_text());missing=sorted({x[k] for x in rows for k in ['source_SMILES','SMILES']}-cache.keys());print('New chemical notations:',len(missing),flush=True)
 with ProcessPoolExecutor(max_workers=workers(64)) as pool:hashes=list(pool.map(identity,missing,chunksize=32))
 cache.update(zip(missing,hashes));write_json(DATA/'molecule_identities.json',cache);report,details=validate(DATA,workers(96));write_json(ROOT/'validation_details.json',details)
 train=json.loads((DATA/'train.json').read_text());trainids={x['sample_id'] for x in train};c={f:collections.Counter() for f in ['atoms','charges','bonds']}
 for d in details:
  if d['sample_id'] not in trainids:continue
  for f,k in KEYS:c[f][k]+=d['subclass_counts'][f].get(k,0)
 expected=json.loads((DATA/'distribution_report.json').read_text())['after']['counts'];assert {f:dict(x) for f,x in c.items()}==expected
 report['independent_subclass_count_check']='passed';report['train_sampling_repeats']=1;write_json(DATA/'validation_report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['review_flags']},indent=2),flush=True)
