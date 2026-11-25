"""Positive-determinant SVG view augmentation with native owner IDs and rebuilt GT."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,copy,io,random,math,json,xml.etree.ElementTree as ET
import numpy as np,cairosvg
from PIL import Image
from runtime import BASE,TOOL
sys.path.insert(0,str(TOOL))
from chem_prep.svg import svg_geometry,matrix,make_bands,raster_paths
from chem_prep.ownership import glyph_mask
from chem_prep.band_audit import StoredBond
from chem_prep.lines import attach,extend_paths,write_artifacts
from chem_prep.visuals import overlay,review
from chem_prep.io import write_json,seed_for
from chem_prep.augmentation import PALETTES,parse_rgb

def points(p,m):return np.column_stack((p,np.ones(len(p))))@m[:2].T

def box(b,m):
 x1,y1,x2,y2=b;p=points(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]),m);return np.r_[p.min(0),p.max(0)].tolist()

def mattext(m):return 'matrix('+','.join(f'{v:.14g}' for v in [m[0,0],m[1,0],m[0,1],m[1,1],m[0,2],m[1,2]])+')'

def bounds(ann):
 allboxes=[a['bbox'] for a in ann['atoms']]+[q['bbox'] for q in ann['charges']]+[b['mask_bbox'] for b in ann['bands']]
 a=np.array(allboxes);return np.r_[a[:,:2].min(0),a[:,2:].max(0)]

def affine(ann,rng):
 w,h=ann['width'],ann['height'];bb=bounds(ann);center=(bb[:2]+bb[2:])/2;angle=rng.uniform(-7,7)*math.pi/180;rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
 corners=np.array([[bb[0],bb[1]],[bb[2],bb[1]],[bb[2],bb[3]],[bb[0],bb[3]]]);pr=(corners-center)@rot.T;extent=np.ptp(pr,axis=0);s=min(rng.uniform(.88,1.14),(w-24)/max(1,extent[0]),(h-24)/max(1,extent[1]));pr*=s
 lo=np.array([12.,12.])-pr.min(0);hi=np.array([w-12.,h-12.])-pr.max(0);target=np.array([rng.uniform(lo[i],max(lo[i],hi[i])) for i in range(2)]);m=np.eye(3);m[:2,:2]=s*rot;m[:2,2]=target-s*rot@center;return m

def make_svg(svg,m,size,tint=None):
 root=ET.fromstring(svg)
 def walk(el,parent):
  if el.tag.endswith('defs'):return
  total=parent@matrix(el.get('transform'));el.attrib.pop('transform',None)
  if el.tag.endswith(('path','use','rect','circle','line','polygon','polyline','ellipse')):el.set('transform',mattext(total))
  if el.get('data-role')=='charge':el.set('data-owner-atom-id',el.get('data-atom-id'))
  for child in el:walk(child,total)
 walk(root,m)
 w,h=size;root.set('width',str(w));root.set('height',str(h));root.set('viewBox',f'0 0 {w} {h}')
 if tint is not None:
  value='rgb('+','.join(f'{v*100:.6f}%' for v in tint)+')'
  for el in root.iter():
   for k in ['fill','stroke']:
    old=el.get(k)
    if old and old not in ['none','white'] and (old in ['black','#000','#000000'] or ((rgb:=parse_rgb(old)) is not None and rgb.min()<.99)):el.set(k,value)
   sty=el.get('style')
   if sty:
    vals=dict(x.strip().split(':',1) for x in sty.split(';') if ':' in x)
    for k in ['fill','stroke']:
     old=vals.get(k)
     if old and old not in ['none','white'] and (old in ['black','#000','#000000'] or ((rgb:=parse_rgb(old)) is not None and rgb.min()<.99)):vals[k]=value
    el.set('style',';'.join(k+':'+v for k,v in vals.items())+';')
 # This full-frame white background precedes every inherited scene object.
 root.insert(0,ET.Element('{http://www.w3.org/2000/svg}rect',{'width':str(w),'height':str(h),'fill':'white'}))
 return ET.tostring(root,encoding='unicode')

def generate(task):
 proto,root_string,view_index=task;root=Path(root_string);rng=random.Random(seed_for(20261120,proto['sample_id'],view_index));source=read_json(root/proto['annotation']);a=copy.deepcopy(source);w,h=a['width'],a['height'];scene=affine(a,rng);full=scene@np.array(a['svg_to_image_matrix']);is_color=rng.random()<.075;tint=rng.choice(PALETTES)[1] if is_color else None
 try:
  svg=make_svg((root/proto['svg']).read_text(),full,(w,h),tint);g,p=svg_geometry(svg);rgb=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGB'))
  for atom in a['atoms']:
   atom['center']=points(np.array([atom['center']]),scene)[0].tolist();atom['bbox']=box(atom['bbox'],scene)
  for q in a['charges']:q['bbox']=box(q['bbox'],scene)
  owners={j:v['native_bond_id'] for j,v in enumerate(p) if v.get('native_bond_id') is not None};qpaths={q['atom_id']:[] for q in a['charges']}
  for j,path in enumerate(p):
   if path.get('native_bond_id') is None and path.get('native_atom_id') in qpaths:qpaths[path['native_atom_id']].append(j)
  excluded={j for ids in qpaths.values() for j in ids};coords=np.array([x['center'] for x in a['atoms']]);bands,masks,unassigned=make_bands(p,coords,[StoredBond(b) for b in a['bonds']],excluded,np.eye(3),(w,h),path_owners=owners)
  for q in a['charges']:
   ids=qpaths[q['atom_id']]
   if not ids:raise ValueError('Missing transformed charge owner')
   ink,bb=raster_paths([p[j] for j in ids],np.eye(3),(w,h));ys,xs=np.nonzero(ink);ib=[int(xs.min()+bb[0]),int(ys.min()+bb[1]),int(xs.max()+bb[0]+1),int(ys.max()+bb[1]+1)]
   q['bbox']=[max(0,ib[0]-1),max(0,ib[1]-1),min(w-1,ib[2]+1),min(h-1,ib[3]+1)];q['ink_bbox']=ib;q['bbox_kind']='transformed_owned_charge_ink_bounds'
  for atom in a['atoms']:
   indices=[j for j,v in enumerate(g) if v.get('native_atom_id')==atom['id']]
   if not indices:continue
   ink,bb=glyph_mask(svg,indices,np.eye(3),(w,h))
   if bb is None:raise ValueError('Empty transformed atom label')
   key='atom_ink_'+str(atom['id']);masks[key]=ink;atom.update(mask_key=key,mask_bbox=bb,ink_area=int(ink.sum()),svg_glyph_ids=[g[j]['sequence'] for j in indices],bbox=[max(0,bb[0]-1),max(0,bb[1]-1),min(w-1,bb[2]+1),min(h-1,bb[3]+1)])
  a['bands']=bands
  for charge in a['charges']:
   x1,y1,x2,y2=charge['bbox'];delta=max(0.,6-(y2-y1))/2;charge['bbox']=[x1,max(0.,y1-delta),x2,min(float(h),y2+delta)]
  for bond,band in zip(a['bonds'],bands):
   bond['endpoints']=[a['atoms'][bond['source']]['center'],a['atoms'][bond['target']]['center']];quad=np.array(band['quad']);bond.update(quad=band['quad'],region_polygon=band['region_polygon'],bbox=np.r_[quad.min(0),quad.max(0)].tolist())
  a['graph']['coords']=[atom['center'] for atom in a['atoms']];a['svg_to_image_matrix']=np.eye(3).tolist()
  stem=proto['image_id']+f'_view{view_index:04d}';paths={k:f'assets/class_aug/{folder}/{stem}.{ext}' for k,folder,ext in [('image','images','png'),('annotation','annotations','json'),('masks','masks','npz'),('svg','svg','svg'),('band_target','targets','json'),('overlay','overlays','png'),('view_check','view_check','jpg')]};paths.update(extend_paths(paths));sample='class_aug/'+stem
  a.update(sample_id=sample,image_id=stem,image=paths['image'],svg=paths['svg'],masks=paths['masks'],band_target=paths['band_target']);a['geometric_augmentation']={'matrix_from_prototype':scene.tolist(),'orientation_preserving':True,'source_prototype':proto['sample_id'],'seed':seed_for(20261120,proto['sample_id'],view_index),'ink_masks_rebuilt_from_transformed_SVG':True,'color_tint':tint}
  a['style']['geometric_augmentation']=a['geometric_augmentation'];attach(a)
  for path in paths.values():(root/path).parent.mkdir(parents=True,exist_ok=True)
  Image.fromarray(rgb).save(root/paths['image']);overlay(rgb,a,masks).save(root/paths['overlay']);review(None,rgb,Image.open(root/paths['overlay']),sample).save(root/paths['view_check'],quality=90);write_artifacts(root,a,paths);(root/paths['svg']).write_text(svg);np.savez_compressed(root/paths['masks'],**masks);write_json(root/paths['annotation'],a);write_json(root/paths['band_target'],{'schema':'independent_band_quad_v1','image':paths['image'],'size':[w,h],'labels':[b['category_id']-1 for b in bands],'quads':[b['quad'] for b in bands]})
  return {**proto,**paths,'sample_id':sample,'image_id':stem,'variant':'class_augmented','is_color':bool(is_color or proto.get('is_color')),'parent_id':proto['original_parent_id'],'status':'ok','prototype_id':proto['sample_id']}
 except Exception as exc:return {'status':'error','prototype_id':proto['sample_id'],'view_index':view_index,'error':str(exc)}

def read_json(path):return json.loads(Path(path).read_text())
