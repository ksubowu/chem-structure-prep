"""Audit stored SVG ownership and export candidate corrections without overwriting data.

Requires no layout regeneration: graph anchors and the exact stored SVG are used.
Changed assignments remain review candidates, not asserted ground truth.
"""
import copy,csv,hashlib,html,json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .svg import svg_geometry,assign_bond_paths,make_bands,raster_paths
from .io import write_json

class AtomAnchor:
    def __init__(self,index):self.i=index
    def index(self):return self.i

class StoredBond:
    def __init__(self,bond):self.b=bond
    def index(self):return self.b['id']
    def source(self):return AtomAnchor(self.b['source'])
    def destination(self):return AtomAnchor(self.b['target'])
    def bondOrder(self):return self.b['order']
    def bondStereo(self):return self.b.get('stereo',0)


def annotation_geometry(ann):
    affine=np.array(ann['svg_to_image_matrix'],float)
    coords=np.array([a['center'] for a in ann['atoms']])
    coords=np.column_stack((coords,np.ones(len(coords))))@np.linalg.inv(affine)[:2].T
    return coords,[StoredBond(b) for b in ann['bonds']]


def _audit(task):
    source,ann_path,out=map(Path,task)
    try:
        a=json.loads(ann_path.read_text());coords,bonds=annotation_geometry(a)
        def resolve(key):
            p=Path(a[key]);return p if p.is_absolute() else source/p
        svg_path=resolve('svg');_,paths=svg_geometry(svg_path.read_text())
        exclude=set()
        # Sign/magnitude ownership uses the stored original charge geometry.
        # This also supports added font-outline charge paths in augmented SVGs.
        for j,p in enumerate(paths):
            if p['style'].get('stroke','none')!='none':continue
            center=(p['bbox'][:2]+p['bbox'][2:])/2
            rendered=np.r_[center,1]@np.array(a['svg_to_image_matrix'])[:2].T
            for q in a.get('charges',[]):
                x1,y1,x2,y2=q['bbox']
                if x1<=rendered[0]<=x2 and y1<=rendered[1]<=y2:exclude.add(j)
        previous,_=assign_bond_paths(paths,coords,bonds,exclude,_angular_weight=0)
        if all(b.get('primitive_assignment')=='native_exclusive_highlight' for b in a['bands']):
            current=[[] for _ in bonds]
            for p in paths:
                if p.get('native_bond_id') is not None:current[p['native_bond_id']].append(p)
            if any(not g for g in current):raise ValueError('Native tagged annotation has a missing bond owner')
        else:current,_=assign_bond_paths(paths,coords,bonds,exclude)
        def ids(group):return sorted(p['sequence'] for p in group)
        # Newer annotations record actual primitive IDs, so repeat audits are idempotent.
        before=[b.get('svg_primitive_ids',ids(g)) for b,g in zip(a['bands'],previous)]
        changed=[i for i,(old,g) in enumerate(zip(before,current)) if sorted(old)!=ids(g)]
        result={'annotation':str(ann_path),'dataset':a.get('dataset',''),'image_id':a.get('image_id',ann_path.stem),'changed_bands':changed,'bonds':len(bonds)}
        if not changed:return result
        # Verify reconstructed historical ownership against saved masks, rather
        # than assuming that legacy metadata always used the same algorithm.
        if not all('svg_primitive_ids' in b for b in a['bands']):
            with np.load(resolve('masks')) as stored:
                for i in changed:
                    ink,bb=raster_paths(previous[i],np.array(a['svg_to_image_matrix']),(a['width'],a['height']))
                    if bb!=a['bands'][i]['mask_bbox'] or not np.array_equal(ink,stored[a['bands'][i]['mask_key']]):
                        raise ValueError(f'Historical ownership reconstruction disagrees with stored B{i} mask; manual review needed')
        bands,masks,unassigned=make_bands(paths,coords,bonds,exclude,np.array(a['svg_to_image_matrix']),(a['width'],a['height']))
        result['primitive_transfers']=[{'band_id':i,'before':before[i],'after':ids(current[i])} for i in changed]
        # Branch by source annotation hash to avoid collisions across BW/color/charge.
        tag=hashlib.sha256(str(ann_path).encode()).hexdigest()[:10];stem=ann_path.stem+'-'+tag
        dest=out/'assets'/stem;dest.mkdir(parents=True,exist_ok=True)
        result['review']='assets/'+stem+'/view_check.jpg';result['corrected_annotation']=str(dest/'annotation.json')
        a=copy.deepcopy(a);a['bands']=bands
        for b,band in zip(a['bonds'],bands):
            p=np.array(band['quad']);b.update(band_id=band['id'],quad=band['quad'],region_polygon=band['region_polygon'],bbox=np.r_[p.min(0),p.max(0)].tolist(),bbox_kind='actual_band_axis_aligned_envelope')
        image=resolve('image');a.update(image=str(image),svg=str(svg_path),masks=str(dest/'masks.npz'),band_target=str(dest/'target.json'))
        a['band_repair']={'revision':2,'original_annotation':str(ann_path),'changed_bands':changed,'review_required':True}
        write_json(dest/'annotation.json',a);np.savez_compressed(dest/'masks.npz',**masks)
        write_json(dest/'target.json',{'schema':'independent_band_quad_v1','image':str(image),'size':[a['width'],a['height']],'labels':[b['category_id']-1 for b in bands],'quads':[b['quad'] for b in bands]})
        from .visuals import overlay
        rgb=np.array(Image.open(image).convert('RGB'));marked=overlay(rgb,a,masks);marked.save(dest/'overlay.png')
        for ann,im in [(json.loads(ann_path.read_text()),Image.open(image).convert('RGB')),(a,marked)]:
            # Draw only affected bands with their bond IDs; do not confuse atom IDs.
            draw=ImageDraw.Draw(im)
            for i in changed:
                q=np.array(ann['bands'][i]['quad']);poly=[tuple(p) for p in q];draw.line(poly+[poly[0]],fill='#e02040',width=3)
                x,y=q.mean(0);draw.rectangle((x,y,x+36,y+15),fill='white');draw.text((x+2,y+1),'B'+str(i),fill='#c00020')
            if ann is a:new=im
            else:old=im
        panels=[(Image.open(image).convert('RGB'),'Indigo image'),(old,'Before: affected bands'),(new,'After: affected bands')]
        canvas=Image.new('RGB',(1800,560),'white');d=ImageDraw.Draw(canvas);d.text((10,8),a.get('image_id',stem),fill='black')
        for i,(im,label) in enumerate(panels):
            im.thumbnail((580,480));canvas.paste(im,(i*600+(600-im.width)//2,55+(480-im.height)//2));d.text((i*600+12,32),label,fill='black')
        canvas.save(dest/'view_check.jpg',quality=92)
        result['needs_extra_review']=[b['id'] for b in bands if b['assignment_max_cost']>.8]
        return result
    except Exception as exc:return {'annotation':str(ann_path),'error':f'{type(exc).__name__}: {exc}'}


def audit(manifests,output,workers=4):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Select a new audit output directory: '+str(out))
    out.mkdir(parents=True);tasks={}
    for manifest in manifests:
        manifest=Path(manifest).resolve()
        rows=json.loads(manifest.read_text()) if manifest.suffix=='.json' else list(csv.DictReader(manifest.open()))
        for r in rows:
            if r.get('status','ok')!='ok':continue
            path=Path(r.get('annotation_file') or r['annotation'])
            if not path.is_absolute():path=manifest.parent/path
            path=path.resolve();tasks[str(path)]=(str(manifest.parent),str(path),str(out))
    if workers==1:results=list(map(_audit,tasks.values()))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:results=list(pool.map(_audit,tasks.values(),chunksize=16))
    changed=[r for r in results if r.get('changed_bands')];errors=[r for r in results if 'error' in r]
    report={'assignment_revision':2,'annotations_checked':len(results),'changed_images':len(changed),'changed_bands':sum(len(r['changed_bands']) for r in changed),'failures':errors,'source_files_modified':False,'note':'Assignment changes are review candidates. Original images, annotations, train/val lists are untouched.','changed':changed}
    write_json(out/'report.json',report)
    entries=[]
    for r in changed:entries.append('<article><h3>'+html.escape(r['dataset']+'/'+r['image_id'])+'</h3><p>Bands '+str(r['changed_bands'])+'</p><img loading="lazy" width="100%" src="'+html.escape(r['review'])+'"></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Band assignment audit</title><h1>Band assignment changes</h1><p>Left: image; middle: before; right: candidate correction. Review required.</p>'+''.join(entries))
    return {k:v for k,v in report.items() if k!='changed'}
