"""Abbreviations that are element symbols must remain opaque dummy labels."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem_prep.backend import create_backend
from chem_prep.augmentation import prepare,hydrogen_count
class AliasElementTests(unittest.TestCase):
 def test_restoring_hydrogen_preserves_stereochemistry(self):
  for sm in ['N[C@@H](C)C(=O)O','C[C@H](O)[C@@H](F)Cl','C[C@H]1CCCCO1','[2H][C@](F)(Cl)C']:
   for seed in range(5):
    with self.subTest(smiles=sm,seed=seed):
     ig,_=create_backend();m=ig.loadMolecule(sm);original=m.canonicalSmiles()
     out,info,_=prepare(ig,m,{'seed':seed,'probability':1,'labels':['Ac']})
     restored=out.clone();restored.getAtom(info['dummy_atom_id']).resetAtom('H');restored.foldHydrogens()
     self.assertEqual(restored.canonicalSmiles(),original)
 def test_aliases_do_not_become_real_elements(self):
  for label in ['Ac','Pr','Ar','CF3','Boc']:
   with self.subTest(label=label):
    ig,_=create_backend();m=ig.loadMolecule('NCCO');before=hydrogen_count(m)
    out,info,_=prepare(ig,m,{'seed':17,'probability':1,'labels':[label]})
    self.assertTrue(info['applied']);a=out.getAtom(info['dummy_atom_id']);self.assertTrue(a.isPseudoatom());self.assertEqual(a.symbol(),label);self.assertEqual(a.degree(),1);self.assertEqual(hydrogen_count(out),before-1)
    reread=ig.loadMolecule(out.smiles());self.assertEqual(sum(a.isPseudoatom() and a.symbol()==label for a in reread.iterateAtoms()),1)
if __name__=='__main__':unittest.main()
