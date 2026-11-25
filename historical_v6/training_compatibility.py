"""Use the current training adapter to check real generated examples."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from runtime import ROOT,DATA
import json,sys,collections
sys.path.insert(0,os.environ.get('CHEM_TRAINING_PROJECT','/mnt/hdd2/bo/work/cvs/portable_mp123_human_band_transfer_20261110/project'))
import torch
torch.set_num_threads(1)
from band_training.mixed_data import prepare_annotation
from band_training.model import targets_for
from band_training.full_v3_data import scaled_lines
from band_training.scored_bond_model import scored_targets,scored_bond_loss
if __name__=='__main__':
 rows=json.loads((DATA/'train.json').read_text());stride=max(1,len(rows)//512);picks=rows[::stride][:512];counts=collections.Counter()
 for row in picks:
  a=json.loads((DATA/row['annotation']).read_text());b=json.loads((DATA/row['band_target']).read_text());l=json.loads((DATA/row['line_target']).read_text());meta,box,labels,quads=prepare_annotation(a,b);heat,entries=targets_for(labels,quads,'cpu',resolve_collisions=True);lines=scaled_lines(a,b,l);scored=scored_targets(labels,quads,lines)
  assert len(entries)==len(labels)==len(a['bonds'])==len(scored[1]);assert len(box['labels'])==len(a['atoms'])+len(a['charges']);counts.update(box['labels'].tolist())
 report={'checked_images':len(picks),'status':'passed','detector_counts':dict(counts)};(ROOT/'training_compatibility.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
