"""Stereochemistry and annotation invariants for fixed-coordinate depictions."""
import sys,json,tempfile,unittest
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem_prep.backend import create_backend
from chem_prep.render import render_one
from chem_prep.lines import extend_paths
from chem_prep.validate import check_one

class FixedLayoutTests(unittest.TestCase):
 def test_two_wedges_preserve_chemistry_and_native_ink_ownership(self):
  ig,_=create_backend();ig.setOption('ignore-stereochemistry-errors',False)
  m=ig.loadMolecule('F[C@](Cl)(Br)I');m.layout();original=m.canonicalSmiles();lines=m.molfile().splitlines();start=4+m.countAtoms();found=None
  for b in m.iterateBonds():
   if b.bondOrder()!=1 or b.bondStereo():continue
   center=next(a.index() for a in m.iterateAtoms() if a.symbol()=='C');s,t=b.source().index(),b.destination().index();other=t if s==center else s
   for direction in [1,6]:
    ls=lines.copy();old=ls[start+b.index()];ls[start+b.index()]=f'{center+1:3}{other+1:3}{1:3}{direction:3}'+old[12:]
    try:
     n=ig.loadMolecule('\n'.join(ls)+'\n')
     if n.canonicalSmiles()==original and {v.bondStereo() for v in n.iterateBonds()}>={5,6}:found=n.molfile()
    except Exception:pass
  self.assertIsNotNone(found)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);paths={k:f'assets/{k}/test.{ext}' for k,ext in [('image','png'),('annotation','json'),('masks','npz'),('svg','svg'),('band_target','json'),('overlay','png'),('view_check','jpg')]};paths.update(extend_paths(paths))
   job={'root':tmp,'paths':paths,'render':{'canvas_size':640,'padding':200,'preserve_source_coordinates':True,'font_families':['DejaVu Sans']},'source':{'dataset':'TEST','id':'test','smiles':'F[C@](Cl)(Br)I','molfile':found},'seed':7,'is_color':False,'dummy_probability':0,'labels':[],'sample_id':'test','variant':'test','fingerprint':'fixed'}
   row=render_one(job);self.assertEqual(row['status'],'ok',row);a=json.loads((root/row['annotation']).read_text());self.assertEqual(Counter(b['category_id'] for b in a['bands']),{1:2,5:1,6:1});self.assertEqual(ig.loadMolecule(a['SMILES']).canonicalSmiles(),original);self.assertEqual(len(a['lines']),4);self.assertNotIn('error',check_one((row,tmp)))

if __name__=='__main__':unittest.main()
