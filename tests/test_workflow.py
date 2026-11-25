"""Scientific invariants, portable IO, repeatability, resume and frozen holdout."""
import copy,csv,json,os,shutil,subprocess,sys,tempfile,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chem_prep.backend import create_backend,set_font,doctor
from chem_prep.augmentation import prepare,hydrogen_count
from chem_prep.balance import make_epoch
from chem_prep.config import load,ingest
from chem_prep.io import read_json,digest
from chem_prep.pipeline import run
from chem_prep.split import identity,assign
from chem_prep.svg import svg_geometry,charge_and_atom_glyphs,make_bands

class ScientificTests(unittest.TestCase):
    def test_backend_returns_pixel_coords(self):self.assertTrue(doctor()['pixel_coords_verified'])

    def test_hydrogen_replacement_preserves_exact_counts_and_isotopes(self):
        for smiles in ['C','C[C@H](O)C(=O)N','[H]C(=O)N','[NH4+]','c1ccccc1','c1ccncc1','c1cc[nH]c1','[C@H](F)(Cl)Br','[2H]C([H])([H])[H]']:
            with self.subTest(smiles=smiles):
                ig,r=create_backend();source=ig.loadMolecule(smiles);before=hydrogen_count(source)
                mol,aug,color=prepare(ig,source,{'seed':17,'probability':1,'labels':['CF3']})
                self.assertTrue(aug['applied']);self.assertEqual(hydrogen_count(mol),before-1)
                self.assertEqual([a.index() for a in mol.iterateAtoms()],list(range(mol.countAtoms())))
                self.assertTrue(mol.getAtom(aug['dummy_atom_id']).isPseudoatom())
                self.assertEqual(hydrogen_count(ig.loadMolecule(mol.smiles())),before-1)

    def test_no_hydrogen_is_not_invented(self):
        ig,r=create_backend();mol,aug,color=prepare(ig,ig.loadMolecule('C(F)(F)(F)F'),{'seed':1,'probability':1,'labels':['CF3']})
        self.assertFalse(aug['applied']);self.assertEqual(aug['reason'],'no_eligible_nonisotopic_bonded_H')

    def test_short_double_bond_axis_is_not_rotated(self):
        ig,r=create_backend();ig.setOption('render-output-format','svg');ig.setOption('render-bond-length',45);ig.setOption('render-relative-thickness',2.0);ig.setOption('render-label-mode','hetero');set_font(ig,'DejaVu Sans')
        mol=ig.loadMolecule('NC(CC(C1=CC=CC=C1)C2=CC=CC=C2)=O');mol.layout();svg=bytes(r.renderToBuffer(mol)).decode()
        import xml.etree.ElementTree as ET
        vb=ET.fromstring(svg).get('viewBox').split();coords=np.array([a.coords() for a in mol.iterateAtoms()]);g,p=svg_geometry(svg);_,charges,excluded=charge_and_atom_glyphs(g,p,coords,[a.charge() or 0 for a in mol.iterateAtoms()])
        bonds=list(mol.iterateBonds());bands,masks,_=make_bands(p,coords,bonds,excluded,np.eye(3),(int(float(vb[2])),int(float(vb[3]))))
        b=next(b for b in bonds if b.bondOrder()==2 and 'O' in [b.source().symbol(),b.destination().symbol()]);band=bands[b.index()]
        self.assertEqual(band['svg_primitive_count'],2)
        axis=coords[b.destination().index()]-coords[b.source().index()];q=np.array(band['quad']);qa=q[2:].mean(0)-q[:2].mean(0)
        self.assertGreater(abs(np.dot(axis,qa)/(np.linalg.norm(axis)*np.linalg.norm(qa))),.98)

    def test_bridged_carbonyl_strokes_do_not_enter_single_bond(self):
        from chem_prep.band_audit import annotation_geometry
        from chem_prep.svg import assign_bond_paths
        fixture=ROOT/'tests/fixtures'
        ann=read_json(fixture/'bridged_carbonyl.json');coords,bonds=annotation_geometry(ann)
        _,paths=svg_geometry((fixture/'bridged_carbonyl.svg').read_text())
        old,_=assign_bond_paths(paths,coords,bonds,set(),_angular_weight=0)
        self.assertEqual([len(old[i]) for i in (0,7)],[1,2])
        groups,_=assign_bond_paths(paths,coords,bonds,set())
        self.assertEqual([next(i for i,q in enumerate(paths) if q is p) for p in groups[0]],[0,1])
        self.assertEqual([next(i for i,q in enumerate(paths) if q is p) for p in groups[7]],[8])
        bands,masks,_=make_bands(paths,coords,bonds,set(),np.array(ann['svg_to_image_matrix']),(ann['width'],ann['height']))
        self.assertEqual(bands[0]['svg_primitive_count'],2)
        self.assertEqual(bands[7]['svg_primitive_count'],1)
        # Positive regression: every original carbonyl-stroke pixel belongs to B0.
        from chem_prep.svg import raster_paths
        expected,bb=raster_paths(paths[:2],np.array(ann['svg_to_image_matrix']),(ann['width'],ann['height']))
        self.assertEqual(bb,bands[0]['mask_bbox'])
        np.testing.assert_array_equal(expected,masks['ink_0'])

    def test_reference_style_and_long_dummy_label(self):
        from PIL import Image,ImageDraw
        from chem_prep.pipeline import render
        from chem_prep.validate import audit_assets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);im=Image.new('RGB',(250,140),'white');d=ImageDraw.Draw(im);d.line([(20,110),(100,30),(210,90)],fill='black',width=3);im.save(root/'ref.png')
            (root/'input.csv').write_text('image_id,SMILES,file_path\n1,CCO,ref.png\n')
            (root/'labels.json').write_text(json.dumps(['OCH2CHOHCH2OH']))
            (root/'config.yaml').write_text(yaml.safe_dump({'inputs':[{'csv':'input.csv','image_column':'file_path'}],'output':'out','workers':1,'render':{'canvas_size':[480,320]},'color':{'count':1,'dummy_probability':1,'labels_file':'labels.json'}}))
            cfg=load(root/'config.yaml');rows=render(cfg);report,_=audit_assets(root/'out',rows,1);self.assertFalse(report['failures'])
            colored=next(r for r in rows if r['is_color']);ann=read_json(root/'out'/colored['annotation']);self.assertTrue(ann['style']['reference_matched']);self.assertEqual((ann['width'],ann['height']),(480,320))
            self.assertEqual(ann['augmentation']['label_glyph_count'],len('OCH2CHOHCH2OH'))
            ig,r=create_backend();self.assertEqual(ig.loadMolecule(ann['rendered_molfile']).countAtoms(),len(ann['atoms']))

    def test_canonicalization_and_unsplittable_group(self):
        self.assertEqual(identity('c1ccccc1'),identity('C1=CC=CC=C1'))
        rows=[{'sample_id':str(i),'dataset':'same','variant':'bw','dummy_applied':False,'has_charge':False,'source_SMILES':'CCO','SMILES':'CCO'} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            train,val,report=assign(Path(tmp),rows,{'validation_fraction':.5,'stratify':['dataset'],'seed':1,'solver_seconds':2})
            self.assertEqual(len(train),2);self.assertEqual(len(val),0);self.assertFalse(report['exact_stratum_counts'])

    def test_balancer_preserves_unique_assets_and_reports_incompatible_color_mix(self):
        rows=[{'sample_id':'positive','has_charge':True,'is_color':True,'dummy_applied':False},{'sample_id':'negative','has_charge':False,'is_color':False,'dummy_applied':False}]
        epoch,report=make_epoch(rows,rows,.25,17)
        self.assertEqual(Fraction(sum(r['has_charge'] for r in epoch),len(epoch)),Fraction(1,4));self.assertEqual({r['asset_sample_id'] for r in epoch},{'positive','negative'})
        self.assertTrue(report['color_mix_relaxed'])
        with self.assertRaisesRegex(ValueError,'both charged and uncharged'):make_epoch(rows[:1],rows[:1],.25,17)

    def test_csv_schema_quarantine_and_optional_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input.csv').write_text('id,smiles\n1,CCO\n2,\n1,CCN\n')
            (root/'config.yaml').write_text(yaml.safe_dump({'inputs':[{'csv':'input.csv','id_column':'id','smiles_column':'smiles'}],'output':'out'}))
            records,issues,files=ingest(load(root/'config.yaml'));self.assertEqual(len(records),1);self.assertEqual(len(issues),2)

class IntegrationTests(unittest.TestCase):
    def test_full_workflow_reproducibility_resume_frozen_val_and_relocation(self):
        with tempfile.TemporaryDirectory(prefix='chem-prep-test-') as tmp:
            work=Path(tmp);shutil.copy2(ROOT/'chem_prep/assets/example/demo.csv',work/'demo.csv')
            raw=yaml.safe_load((ROOT/'chem_prep/assets/example/recipe.yaml').read_text());raw['inputs'][0]['csv']=str(work/'demo.csv');raw['workers']=1;raw['render']={'canvas_size':384};raw['charge_augmentation']['canvas_size']=512;raw['charge_augmentation']['count']=8
            raw['output']=str(work/'a');(work/'a.yaml').write_text(yaml.safe_dump(raw));a=load(work/'a.yaml');report=run(a)
            self.assertFalse(report['failures']);self.assertEqual(report['train']['charged_fraction'],.25)
            root=work/'a';first_val=digest(root/'val.json');first_train=digest(root/'train.json');assets=read_json(root/'data_manifest.json')
            self.assertTrue(any(r['dummy_applied'] for r in assets));self.assertTrue(any(2 in r['charge_values'] for r in assets))
            raw['workers']=2;raw['output']=str(work/'b');(work/'b.yaml').write_text(yaml.safe_dump(raw));run(load(work/'b.yaml'))
            assets_b=read_json(work/'b/data_manifest.json');self.assertEqual({r['sample_id']:digest(root/r['image']) for r in assets},{r['sample_id']:digest(work/'b'/r['image']) for r in assets_b})
            self.assertEqual(read_json(root/'val.json'),read_json(work/'b/val.json'));self.assertEqual(read_json(root/'train.json'),read_json(work/'b/train.json'))
            # Resume must neither reshuffle the holdout nor append sampling repeats.
            run(a);self.assertEqual(digest(root/'val.json'),first_val);self.assertEqual(digest(root/'train.json'),first_train)
            moved=work/'relocated';shutil.copytree(root,moved)
            env={**os.environ,'PYTHONPATH':str(ROOT),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
            check=subprocess.run([sys.executable,'-m','chem_prep','validate','--dataset',str(moved),'--workers','1'],cwd=work,env=env,capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
            change=subprocess.run([sys.executable,'-m','chem_prep','balance','--dataset',str(moved),'--charged-fraction','.3','--workers','1'],cwd=work,env=env,capture_output=True,text=True)
            self.assertEqual(change.returncode,0,change.stderr);self.assertEqual(digest(moved/'val.json'),first_val);self.assertEqual(read_json(moved/'balance_report.json')['actual_charged_fraction'],.3)
            original=(moved/'val.json').read_bytes();(moved/'val.json').write_bytes(original+b' ')
            check=subprocess.run([sys.executable,'-m','chem_prep','validate','--dataset',str(moved),'--workers','1'],cwd=work,env=env,capture_output=True,text=True)
            self.assertNotEqual(check.returncode,0);self.assertIn('Validation changed',check.stderr)

if __name__=='__main__':unittest.main()
