"""Native ownership: complete labels, color collisions, bridge bonds and invariance."""
import sys,unittest,io,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,cairosvg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chem_prep.backend import create_backend,set_font
from chem_prep.ownership import OwnershipProbe,infer_ownership,geometry_check,glyph_mask,annotate_svg
from chem_prep.svg import svg_geometry,charge_and_atom_glyphs,make_bands


def scene(smiles):
    ig,r=create_backend();set_font(ig,'DejaVu Sans');ig.setOption('render-output-format','svg');ig.setOption('render-label-mode','hetero');ig.setOption('render-bond-length',45);ig.setOption('render-implicit-hydrogens-visible',True);ig.setOption('render-coloring',True)
    mol=ig.loadMolecule(smiles);mol.layout();svg=bytes(r.renderToBuffer(mol)).decode();return ig,r,mol,svg

class NativeOwnershipTests(unittest.TestCase):
    def test_nh2_group_and_preexisting_red_oxygen_do_not_collide(self):
        ig,r,mol,_=scene('NC1=CC=CC=C1N1CCNCC1.O')
        oxygen=mol.getAtom(mol.countAtoms()-1);oxygen.highlight();ig.setOption('render-highlight-color','1,0,0');svg=bytes(r.renderToBuffer(mol)).decode()
        owners,bonds,report=infer_ownership(ig,r,mol,svg)
        self.assertTrue(oxygen.isHighlighted());self.assertEqual(ig.getOption('render-highlight-color'),'[1, 0, 0]')
        self.assertEqual([j for j,owner in owners.items() if owner==0],[0,1,2])
        self.assertNotIn(12,owners.values())
        with OwnershipProbe(ig,r,mol,svg) as p:hit=p.select(atom_ids=[0],keep_svg=True)
        rgba=[np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=s.encode())))).astype(int) for s in hit['probe_svgs']]
        changed=np.any(rgba[0]!=rgba[1],axis=2);ys,xs=np.nonzero(changed)
        self.assertGreater(len(xs),0)
        glyphs,_=svg_geometry(svg)
        for j,owner in owners.items():
            if owner!=oxygen.index():continue
            x1,y1,x2,y2=np.floor(glyphs[j]['bbox']).astype(int);self.assertFalse(changed[max(0,y1):y2+1,max(0,x1):x2+1].any())
        self.assertTrue(report['production_colors_restored'])

    def test_group_selection_and_geometry_mismatch_are_explicit(self):
        ig,r,mol,svg=scene('NCCO')
        with OwnershipProbe(ig,r,mol,svg) as p:
            hit=p.select(atom_ids=[0,3]);self.assertGreaterEqual(len(hit['glyph_indices']),4)
            with self.assertRaisesRegex(ValueError,'out of range'):p.select(atom_ids=[99])
        root=ET.fromstring(svg);root.set('viewBox','0 0 999 999')
        with self.assertRaisesRegex(ValueError,'canvas'):geometry_check(svg,ET.tostring(root,encoding='unicode'))

    def test_native_bridge_bond_ownership_and_persistent_ids(self):
        for smiles in ['O=C1CC2CCC(C1)N2CC1=CC=CC=C1','CC(C)(C)OC(=O)N1C2CCC1CC(=O)C2']:
            with self.subTest(smiles=smiles):
                ig,r,mol,svg=scene(smiles);owners,paths,report=infer_ownership(ig,r,mol,svg);tagged=annotate_svg(svg,owners,paths);g,p=svg_geometry(tagged)
                self.assertEqual({j:v['native_bond_id'] for j,v in enumerate(p) if v.get('native_bond_id') is not None},paths)
                coords=np.array([a.coords() for a in mol.iterateAtoms()]);_,_,exclude=charge_and_atom_glyphs(g,p,coords,[a.charge() or 0 for a in mol.iterateAtoms()],glyph_owners=owners)
                size=tuple(map(lambda s:int(float(s)),ET.fromstring(svg).get('viewBox').split()[2:]));bonds=list(mol.iterateBonds());bands,masks,_=make_bands(p,coords,bonds,exclude,np.eye(3),size,path_owners=paths)
                for b,band in zip(bonds,bands):
                    self.assertEqual(band['primitive_assignment'],'native_exclusive_highlight')
                    if b.bondOrder()==2 and 'O' in [b.source().symbol(),b.destination().symbol()]:self.assertEqual(band['svg_primitive_count'],2)

    def test_crowded_bridge_printed_ch2_is_not_treated_as_invisible_carbon(self):
        ig,r,mol,svg=scene('NN=C(O)C12CC3CC(CC(C3)C1)C2')
        owners,paths,report=infer_ownership(ig,r,mol,svg)
        for atom_id in [5,7,11]:self.assertEqual(sum(v==atom_id for v in owners.values()),3)

    def test_discarded_hidden_carbon_probe_cannot_reposition_other_h_labels(self):
        ig,r,mol,svg=scene('C/C=C1\\NC(=O)[C@H]2CSSCC/C=C/[C@H](CC(=O)N[C@H](C(C)C)C(=O)N2)OC(=O)[C@H](C(C)C)NC1=O')
        owners,paths,report=infer_ownership(ig,r,mol,svg)
        self.assertNotIn(1,owners.values());self.assertTrue(report['production_colors_restored'])

    def test_smarts_group_export_keeps_native_matcher_alive(self):
        import tempfile,json
        from chem_prep.probe import export
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'probe'
            result=export('NCCO',out,group_smarts='[NX3;H2]',colored=True)
            self.assertEqual(result['selections'],1)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['selections'][0]['atom_ids'],[0])
            self.assertGreater(report['selections'][0]['ink_area'],0)

    def test_wide_scene_hidden_carbon_trial_float32_translation(self):
        import json
        fixture=json.loads((ROOT/'tests/fixtures/wide_float32_probe.json').read_text())
        ig,r,mol,_=scene(fixture['smiles']);set_font(ig,fixture['font']);ig.setOption('render-relative-thickness',fixture['relative_thickness'])
        svg=bytes(r.renderToBuffer(mol)).decode();owners,paths,report=infer_ownership(ig,r,mol,svg)
        self.assertTrue(report['geometry_verified']);self.assertTrue(report['production_colors_restored'])
        self.assertNotIn(77,owners.values())

    def test_charges_explicit_isotopic_h_and_long_alias(self):
        for smiles in ['[2H]OC[NH2+]C.[O-]C','[NH3+]CC(=O)[O-]','C* |$;OCH2CHOHCH2OH$|']:
            with self.subTest(smiles=smiles):
                ig,r,mol,svg=scene(smiles);owners,paths,report=infer_ownership(ig,r,mol,svg);g,p=svg_geometry(svg);coords=np.array([a.coords() for a in mol.iterateAtoms()]);grouped,charges,excluded=charge_and_atom_glyphs(g,p,coords,[a.charge() or 0 for a in mol.iterateAtoms()],glyph_owners=owners)
                self.assertEqual(len(charges),sum(bool(a.charge()) for a in mol.iterateAtoms()))
                for a in mol.iterateAtoms():
                    if a.isPseudoatom():self.assertEqual(len(grouped[a.index()]),len('OCH2CHOHCH2OH'))
                mask,bb=glyph_mask(svg,list(owners),np.eye(3),(1000,1000));self.assertTrue(mask.any());self.assertIsNotNone(bb)

if __name__=='__main__':unittest.main()
