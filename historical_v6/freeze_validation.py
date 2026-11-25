"""Reproduce the user-selected 1,497-row holdout without editing the source."""
from runtime import ROOT,BASE,DATA,TOOL
from pathlib import Path
import json,sys,shutil,hashlib
sys.path.insert(0,str(TOOL))
from chem_prep.balance import snapshot_validation
from chem_prep.io import write_json
if __name__=='__main__':
 source=ROOT/'validation_inputs'
 if not source.exists():source=BASE/'indigo_full_v3_20261116'
 rows=json.loads((source/'val.json').read_text());assert len(rows)==1497
 for name in ['val.json','val.csv']:
  if (DATA/name).exists() and (DATA/'holdout_snapshot.json').exists():
   if (DATA/name).read_bytes()!=(source/name).read_bytes():raise RuntimeError('Refuse to overwrite the frozen holdout with a different manifest')
  else:shutil.copy2(source/name,DATA/name)
 state=snapshot_validation(DATA);write_json(ROOT/'validation_reference.json',{'source':str(source),'rows':1497,'files':len(state['files']),'manifest_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['val.json','val.csv']},'candidate_exclusion_reference':'Conservative union of frozen 1498 and current 1497'})
