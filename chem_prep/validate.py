"""Independent artifact, geometry, sampling and frozen-validation audits."""
import copy,io,math,os,xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cairosvg,cv2,numpy as np
from PIL import Image
from .io import read_json,write_json
from .balance import verify_holdout
SVG='http://www.w3.org/2000/svg'


def require(condition,message):
    if not condition:raise ValueError(message)


def check_one(task):
    row,root=task;root=Path(root)
    try:
        cv2.setNumThreads(1)
        fields=['image','annotation','masks','band_target','svg','view_check','overlay']
        fields.extend(k for k in ['line_target','line_overlay','band_line_view_check'] if k in row)
        for field in fields:
            path=os.path.abspath(root/row[field]);require(os.path.commonpath([str(root.resolve()),path])==str(root.resolve()),'Artifact path escapes dataset root');require(Path(path).is_file(),'Missing '+field)
        ann=read_json(root/row['annotation']);w,h=ann['width'],ann['height'];atoms=ann['atoms'];n=len(atoms)
        rgb=np.array(Image.open(root/row['image']).convert('RGB'));require(rgb.shape==(h,w,3),'Image size mismatch')
        require([a['id'] for a in atoms]==list(range(n)),'Non-dense atom IDs')
        require(ann['graph']['num_atoms']==n and len(ann['graph']['edges'])==n,'Graph shape mismatch')
        require(all(len(e)==n for e in ann['graph']['edges']),'Non-square adjacency')
        def bbox(box):
            a,b,c,d=box;require(np.isfinite(box).all() and 0<=a<=c<=w and 0<=b<=d<=h,'Invalid bbox '+str(box))
        for a in atoms:
            bbox(a['bbox']);require(0<=a['center'][0]<=w and 0<=a['center'][1]<=h,'Invalid atom anchor')
        expected={a['id']:a['formal_charge'] for a in atoms if a['formal_charge']};actual={q['atom_id']:q['value'] for q in ann['charges']}
        require(actual==expected,'Missing/wrong charge annotations');require(len(actual)==len(ann['charges']),'Duplicate charge annotation')
        for q in ann['charges']:bbox(q['bbox']);require(q['bbox'][2]>q['bbox'][0] and q['bbox'][3]>q['bbox'][1],'Empty charge box')
        require(bool(ann['charges'])==row['has_charge'],'Manifest charge class disagrees with annotation')
        require(len(ann['bands'])==len(ann['bonds']),'Band/bond count mismatch')
        target=read_json(root/row['band_target']);require(set(target)=={'schema','image','size','labels','quads'},'Band supervision contains unsupported or graph-dependent fields')
        require(target['quads']==[b['quad'] for b in ann['bands']],'Band target geometry mismatch');require(target['labels']==[b['category_id']-1 for b in ann['bands']],'Band target labels mismatch')
        if 'lines' in ann or 'line_target' in row:
            require(all(k in row for k in ['line_target','line_overlay','band_line_view_check']),'Missing line artifacts')
            mode=ann.get('line_endpoint_mode');require(mode in ['atom_center','box_center'],'Wrong line endpoint mode')
            require(len(ann['lines'])==len(ann['bonds']),'Line/bond count mismatch')
            for b,band,line in zip(ann['bonds'],ann['bands'],ann['lines']):
                expected_points=[]
                for idx in [b['source'],b['target']]:
                    atom=atoms[idx]
                    if mode=='atom_center':expected_points.append(atom['center'])
                    else:
                        x1,y1,x2,y2=atom['bbox'];expected_points.append([(x1+x2)/2,(y1+y2)/2])
                require(line['bond_id']==b['id']==line['id']==b['line_id'],'Line/bond identity mismatch')
                require(line['source']==b['source'] and line['target']==b['target'],'Line atom identity mismatch')
                require(line['category_id']==band['category_id'] and line['endpoint_mode']==mode,'Line type/mode mismatch')
                require(line['endpoints']==expected_points and b['line']==expected_points,'Line endpoints disagree with atom/box centers')
            lt=read_json(root/row['line_target'])
            require(set(lt)=={'schema','image','size','endpoint_mode','labels','lines'} and lt['schema']=='independent_bond_line_v1','Invalid line supervision schema')
            require(lt['image']==ann['image'] and lt['size']==[w,h] and lt['endpoint_mode']==mode,'Line target image/mode mismatch')
            require(lt['lines']==[v['endpoints'] for v in ann['lines']] and lt['labels']==target['labels'],'Line target geometry/labels mismatch')
        dark=rgb.min(2)<245;near=cv2.dilate(dark.astype(np.uint8),np.ones((3,3),np.uint8))>0;aligned=area=0;high=0
        with np.load(root/row['masks']) as masks:
            for atom in atoms:
                if 'mask_key' not in atom:continue
                x1,y1,x2,y2=atom['mask_bbox'];ink=masks[atom['mask_key']]
                require(ink.shape==(y2-y1,x2-x1) and ink.dtype==bool,'Atom ink mask shape/type mismatch')
                require(int(ink.sum())==atom['ink_area'],'Atom ink mask area mismatch')
                bb=atom['bbox'];require(bb[0]<=x1 and bb[1]<=y1 and bb[2]>=x2-1 and bb[3]>=y2-1,'Atom box omits owned glyph ink')
                require((ink&near[y1:y2,x1:x2]).sum()/max(1,ink.sum())>.85,'Atom ink/image misalignment')
            for b,band in zip(ann['bonds'],ann['bands']):
                s,t=b['source'],b['target'];require(0<=s<n and 0<=t<n,'Invalid bond indices');require(b['endpoints']==[atoms[s]['center'],atoms[t]['center']],'Wrong bond endpoints')
                q=np.array(band['quad']);require(q.shape==(4,2) and np.isfinite(q).all(),'Invalid quad');require(np.all(q>=[0,0]) and np.all(q<=[w,h]) and cv2.contourArea(q.astype(np.float32))>0,'Out-of-bounds/empty band')
                bbox(band['mask_bbox']);x1,y1,x2,y2=band['mask_bbox'];ink=masks[band['mask_key']];region=masks[band['region_mask_key']]
                require(ink.shape==(y2-y1,x2-x1) and ink.dtype==bool,'Wrong mask shape/type');require(np.all(~ink|region),'Band region omits ink');require(int(ink.sum())==band['ink_area'],'Mask area mismatch')
                aligned+=int((ink&near[y1:y2,x1:x2]).sum());area+=int(ink.sum());high+=band['assignment_max_cost']>.4
        require(not area or aligned/area>.85,'Band mask/image misalignment')
        glyphs_checked=0
        if ann.get('charge_augmentation'):
            tree=ET.fromstring((root/row['svg']).read_text());nodes=[p for p in tree.iter() if p.get('data-role')=='charge'];require(len(nodes)==len(ann['charges']),'Charge glyph/box count mismatch')
            for node,q in zip(nodes,ann['charges']):
                require(int(node.get('data-atom-id'))==q['atom_id'],'Wrong charge owner')
                x1,y1,x2,y2=q['bbox'];left,top=math.floor(x1-2),math.floor(y1-2);ww,hh=math.ceil(x2+2)-left,math.ceil(y2+2)-top
                isolated=ET.Element(f'{{{SVG}}}svg',{'width':str(ww),'height':str(hh),'viewBox':f'{left} {top} {ww} {hh}'});el=copy.deepcopy(node);el.set('fill','black');isolated.append(el)
                alpha=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ET.tostring(isolated)))).convert('RGBA'))[:,:,3];ys,xs=np.where(alpha>0)
                require(len(xs)>0,'Empty charge glyph');require(xs.min()+left>=x1-.51 and xs.max()+left+1<=x2+.51 and ys.min()+top>=y1-.51 and ys.max()+top+1<=y2+.51,'Charge box does not contain glyph ink');glyphs_checked+=1
        subclass_counts={'atoms':dict(Counter(a['symbol'] for a in atoms if not a.get('is_pseudoatom',False))),
                         'charges':dict(Counter(str(q['value']) for q in ann['charges'])),
                         'bonds':dict(Counter(str(b['category_id']) for b in ann['bands']))}
        return {'sample_id':row['sample_id'],'atoms':n,'bonds':len(ann['bonds']),'charges':len(ann['charges']),'area':area,'aligned':aligned,'high_cost_bands':high,'glyphs_checked':glyphs_checked,'subclass_counts':subclass_counts,'review_band_ids':[b['id'] for b in ann['bands'] if b['assignment_max_cost']>.4]}
    except Exception as e:return {'sample_id':row['sample_id'],'error':f'{type(e).__name__}: {e}'}


def audit_assets(root,rows,workers=4):
    tasks=[(r,str(root)) for r in rows]
    if workers==1:results=list(map(check_one,tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:results=list(pool.map(check_one,tasks,chunksize=8))
    errors=[r for r in results if 'error' in r];totals=Counter()
    for r in results:
        if 'error' not in r:totals.update({k:v for k,v in r.items() if isinstance(v,(int,float))})
    report={'records':len(rows),'failures':errors,'review_flags':[{'sample_id':r['sample_id'],'band_ids':r['review_band_ids']} for r in results if 'error' not in r and r['review_band_ids']],**dict(totals),'pixel_alignment_1px':totals['aligned']/max(1,totals['area'])}
    return report,results


def validate(root,workers=4):
    root=Path(root).resolve();assets=read_json(root/'data_manifest.json');report,results=audit_assets(root,assets,workers)
    ids={r['sample_id'] for r in assets};require(len(ids)==len(assets),'Duplicate unique-asset IDs')
    if (root/'train.json').exists():
        train=read_json(root/'train.json');val=read_json(root/'val.json');require({r['group'] for r in train}.isdisjoint({r['group'] for r in val}),'Train/validation group leakage')
        require(len({r['sample_id'] for r in train})==len(train),'Duplicate draw IDs')
        require({r.get('asset_sample_id',r['sample_id']) for r in train}|{r['sample_id'] for r in val}==ids,'Lost/unlisted unique assets')
        identity_cache=read_json(root/'molecule_identities.json');seen={}
        for label,rows in [('train',train),('val',val)]:
            for r in rows:
                for sm in [r['source_SMILES'],r['SMILES']]:
                    key=identity_cache[sm];require(key not in seen or seen[key]==label,'Source/rendered chemistry crosses split');seen[key]=label
        report['train']={'rows':len(train),'unique_images':len({r.get('asset_sample_id',r['sample_id']) for r in train}),'charged_fraction':sum(r['has_charge'] for r in train)/max(1,len(train))}
        report['val']={'rows':len(val),'charged_images':sum(r['has_charge'] for r in val)}
        report['validation_frozen']=verify_holdout(root)
        if (root/'balance_report.json').exists():require(abs(report['train']['charged_fraction']-read_json(root/'balance_report.json')['target_charged_fraction'])<1e-12,'Sampling balance target missed')
    if (root/'render_summary.json').exists():report['render_summary']=read_json(root/'render_summary.json')
    write_json(root/'validation_report.json',report)
    if report['failures']:raise RuntimeError(f'Artifact validation failed for {len(report["failures"])} samples; see validation_report.json')
    return report,results
