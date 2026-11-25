"""Small molecule-disjoint training/calibration/test corpus with audited SVG masks."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_pubchem_bands import prepare, render_record, comparison, write_json, bond_class, CHARGE_CLASSES


def region_quad(ink, category):
    """Ordered tapered region from visible ink only: no atom boxes or indices."""
    ys, xs = np.nonzero(ink)
    pts = np.column_stack((xs,ys)).astype(float)
    center=pts.mean(0)
    _,vectors=np.linalg.eigh(np.cov((pts-center).T))
    axis=vectors[:,-1]
    if axis[0]<0 or (abs(axis[0])<1e-7 and axis[1]<0): axis=-axis
    normal=np.array([-axis[1],axis[0]])
    t=(pts-center)@axis; u=(pts-center)@normal
    lo,hi=t.min(),t.max(); length=hi-lo
    widths=[]
    for a,b in [(lo,lo+.3*length),(hi-.3*length,hi)]:
        us=u[(t>=a)&(t<=b)]
        widths.append(max(1.,float(np.max(np.abs(us)))+.5))
    # Wedge/dash orientation is measured from widening ink, not atom identities.
    if category in (5,6) and widths[0]>widths[1]:
        axis=-axis;normal=-normal;t=-t;u=-u;lo,hi=-hi,-lo;widths=widths[::-1]
    if category not in (5,6): widths=[max(widths)]*2
    a=center+lo*axis;b=center+hi*axis
    quad=np.stack((a+widths[0]*normal,a-widths[0]*normal,
                   b-widths[1]*normal,b+widths[1]*normal))
    return quad.tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=320)
    p.add_argument('--scan',type=int,default=4000);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['images','svg','annotations','masks','comparisons','targets']: (args.output/name).mkdir()
    pool=[];seen=set();reject=[]
    with args.csv.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if index>=args.scan:break
            try:
                mol=prepare(row['SMILES']); canonical=Chem.MolToSmiles(Chem.MolFromSmiles(row['SMILES']))
                if canonical in seen or not 8<=mol.GetNumAtoms()<=45:continue
                if any(a.GetFormalCharge() not in {0,*CHARGE_CLASSES} for a in mol.GetAtoms()):continue
                types={bond_class(b) for b in mol.GetBonds()}
                if 0 in types:continue
                seen.add(canonical);row['_row']=index
                pool.append((row,canonical,types,any(a.GetFormalCharge() for a in mol.GetAtoms())))
            except Exception as e:reject.append({'row':index,'stage':'scan','error':str(e)})
    # Include sparse styles and charged structures, then ordinary source-order rows.
    chosen=[]
    for condition,limit in [(lambda c:3 in c[2],24),(lambda c:5 in c[2] and 6 in c[2],32),
                            (lambda c:c[3],64)]:
        added=0
        for c in pool:
            if condition(c) and c not in chosen:
                chosen.append(c);added+=1
                if added>=limit:break
    ordered=chosen+[c for c in pool if c not in chosen]
    records=[];counts=Counter()
    for row,canonical,types,q in ordered:
        if len(records)>=args.count:break
        stem=f"{args.csv.stem}_row{row['_row']}_cid{row['pubchem_cid']}"
        try:
            ann,image,inks,regions=render_record(row,args.csv.resolve(),args.output,640)
            quads=[region_quad(inks[b['index']],b['category_id']) for b in ann['bonds']]
            # This is the complete input to the band training criterion.
            target={'schema':'independent_band_quad_v1','image':ann['image'],'size':[640,640],
                    'labels':[b['category_id']-1 for b in ann['bonds']], 'quads':quads}
            assert all(np.isfinite(np.asarray(quad)).all() for quad in quads)
            write_json(args.output/'targets'/f'{stem}.json',target)
            record={'image':ann['image'],'truth':f'annotations/{stem}.json',
                    'band_target':f'targets/{stem}.json','canonical_smiles':canonical,
                    'group':hashlib.sha256(canonical.encode()).hexdigest(),'source_row':row['_row']}
            records.append(record)
            counts.update(b['type'] for b in ann['bonds'])
            if len(records)<=4: comparison(ann,image,inks,regions,args.output/'comparisons'/f'{stem}.png')
            print(f"{len(records)}/{args.count} row={row['_row']} atoms={len(ann['atoms'])} bonds={len(quads)}",flush=True)
        except Exception as e:
            reject.append({'row':row['_row'],'stage':'render','error':str(e)})
            for directory in ['images','svg','annotations','masks','comparisons','targets']:
                for path in (args.output/directory).glob(stem+'.*'):path.unlink()
    assert len(records)==args.count, f'Only {len(records)} records passed'
    # Seeded exact split, grouped by canonical molecule, independent of source CSV order.
    rng=np.random.default_rng(20261113);order=rng.permutation(len(records))
    ntest=max(16,len(records)//8);ncal=ntest
    for i,index in enumerate(order):records[index]['split']='test' if i<ntest else 'calibration' if i<ntest+ncal else 'train'
    groups={s:{r['group'] for r in records if r['split']==s} for s in ['train','calibration','test']}
    assert not groups['train']&groups['test'] and not groups['train']&groups['calibration'] and not groups['test']&groups['calibration']
    write_json(args.output/'manifest.json',records)
    report={'status':'passed','count':len(records),'splits':dict(Counter(r['split'] for r in records)),
            'bond_counts':dict(counts),'rejects':reject,'seed':20261113,'source':str(args.csv.resolve()),
            'band_loss_inputs':['image_features','labels','quads'],
            'atom_indices_used_in_band_training':False,'geometry':'PCA/tapered quadrilateral from actual bond ink',
            'limitations':['One rendering style, small pilot; no real-image generalization claim.',
                           'Aromatic rings rendered as Kekule single/double bonds.',
                           'GT atom links only in separate evaluation annotations.']}
    write_json(args.output/'dataset_report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
