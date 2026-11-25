"""Check charge-size guarantees only for newly generated PubChem images."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from runtime import ROOT,DATA,workers
import json,collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def one(row):
 a=json.loads((DATA/row['annotation']).read_text());return [(q['value'],q['bbox'][3]-q['bbox'][1]) for q in a['charges']]
if __name__=='__main__':
 rows=[r for r in json.loads((DATA/'train.json').read_text()) if r['dataset']=='PubChem' and r['has_charge']];vals=collections.defaultdict(list)
 with ProcessPoolExecutor(max_workers=workers(48)) as pool:
  for items in pool.map(one,rows,chunksize=16):
   for value,height in items:vals[str(value)].append(height)
 report={q:{'boxes':len(v),'minimum':float(min(v)),'median':float(np.median(v)),'p10':float(np.quantile(v,.1)),'maximum':float(max(v))} for q,v in vals.items()};assert all(v['minimum']>=5.999 for v in report.values());(ROOT/'charge_box_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
