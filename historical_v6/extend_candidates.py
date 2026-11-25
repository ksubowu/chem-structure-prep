"""Expand genuine Br/B structures rather than repeatedly drawing a few cores."""
from pathlib import Path
import sys,sqlite3,json,collections,hashlib
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent;AUDIT=ROOT.parent/'pubchem_diversity_audit_20261121';sys.path.insert(0,str(AUDIT))
from native_probe import init,probe
from rdkit import Chem,RDLogger
from rdkit.Chem import rdMolHash
RDLogger.DisableLog('rdApp.*')
if __name__=='__main__':
 old=json.loads((AUDIT/'native_probe_results.json').read_text());ids={x['raw_id'] for x in old};db=sqlite3.connect(AUDIT/'inventory.sqlite');db.row_factory=sqlite3.Row;chosen=[]
 for table,col,limit in [('preferred','a_Br',4000),('eligible_known_charges','a_B',10000)]:
  groups=collections.Counter();n=0
  for r in db.execute(f'SELECT s.*,r.cid,r.smiles FROM {table} e JOIN stats s ON s.raw_id=e.raw_id JOIN raw r ON r.id=s.raw_id WHERE s.{col}>0 ORDER BY s.canonical_key'):
   if r['raw_id'] in ids or r['heavy']>80 or set(json.loads(r['charges_all']))-set(['-1','1']):continue
   cl='cyclic:'+r['scaffold_key'].hex() if r['scaffold_key'] else 'acyclic:'+hashlib.blake2b(rdMolHash.MolHash(Chem.MolFromSmiles(r['canonical_smiles']),rdMolHash.HashFunction.AnonymousGraph).encode(),digest_size=16).hexdigest()
   if groups[cl]>=4:continue
   groups[cl]+=1;ids.add(r['raw_id']);n+=1;chosen.append({'raw_id':r['raw_id'],'pubchem_cid':r['cid'],'SMILES':r['smiles'],'scaffold_cluster':cl,'heavy':r['heavy'],'rings':r['rings'],'selection_reasons':[col,'long_tail_diversity'],'source_pool':table})
   if n>=limit:break
  print('Added',col,n,flush=True)
 out=[];batches=[chosen[i:i+128] for i in range(0,len(chosen),128)]
 with ProcessPoolExecutor(max_workers=32,initializer=init) as pool:
  for rows in pool.map(probe,batches):out.extend(rows)
 (ROOT/'native_extra_results.json').write_text(json.dumps(out,indent=2));print('Preflight new real structures',len(out),flush=True)
