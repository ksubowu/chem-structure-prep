"""Copy only the original real-data assets, excluding v5 template expansion."""
from runtime import ROOT,SOURCE,DATA
from pathlib import Path
import json,os,shutil
if __name__=='__main__':
 DATA.mkdir(exist_ok=True);base=json.loads((SOURCE/'source_train.json').read_text());val=json.loads((SOURCE/'val.json').read_text());n=0
 for r in base+val:
  for field in ['image','annotation','svg','masks','band_target','line_target','overlay','line_overlay','view_check','band_line_view_check']:
   target=DATA/r[field];target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():os.link(SOURCE/r[field],target);n+=1
 (ROOT/'base_train.json').write_text(json.dumps(base,indent=2));shutil.copy2(SOURCE/'molecule_identities.json',DATA/'molecule_identities.json');print('Linked original assets',n,'original train',len(base),flush=True)
