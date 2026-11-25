import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from runtime import ROOT,DATA,TOOL
import sys,json,collections
sys.path.insert(0,str(TOOL))
from chem_prep.validate import audit_assets
from chem_prep.io import write_json
if __name__=='__main__':
 rows=[];results=[]
 for p in list((DATA/'_pubchem_checkpoints').glob('*.json'))[:512]:
  x=json.loads(p.read_text());results.append(x)
  if x['status']=='ok':rows.extend(x['rows'])
 report,_=audit_assets(DATA,rows,32);write_json(ROOT/'early_annotation_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='review_flags'},indent=2));print('fallback categories',collections.Counter(e for x in results for e in x.get('fallback_errors',[])));print('typography fallback',sum(bool(x.get('charge_typography_fallback')) for x in results))
 if report['failures']:raise RuntimeError('Early geometry audit failed')
