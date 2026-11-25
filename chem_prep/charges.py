"""Training-only SVG charge-glyph variants with same-molecule, aligned GT."""
import copy,hashlib,io,json,math,random,xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import cairosvg,cv2,numpy as np
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen
from .visuals import overlay,review
from .backend import ASSETS
from .io import asset_stem
from .svg import svg_geometry,charge_and_atom_glyphs,make_bands
from .lines import extend_paths,attach as attach_lines,write_artifacts as write_line_artifacts

SVG='http://www.w3.org/2000/svg';ET.register_namespace('',SVG)
FONTS=[str(ASSETS/'fonts'/name) for name in ['DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSerif.ttf','DejaVuSerif-Bold.ttf','LiberationSans-Regular.ttf','LiberationSans-Italic.ttf','LiberationSerif-Regular.ttf','DejaVuSansMono-Bold.ttf']]
POSITIONS=['upper_right','upper_left','above','right','lower_right']

@lru_cache(maxsize=32)
def font_glyph(font_file,sign):
    font=TTFont(font_file);gs=font.getGlyphSet();cmap=font.getBestCmap()
    code=ord('+') if sign>0 else (0x2212 if 0x2212 in cmap else ord('-'))
    glyph=gs[cmap[code]];pen=SVGPathPen(gs);glyph.draw(pen);bounds=BoundsPen(gs);glyph.draw(bounds)
    return pen.getCommands(),bounds.bounds,chr(code)

class Index:
    def __init__(self,index):self.value=index
    def index(self):return self.value
class Bond:
    def __init__(self,data):self.data=data
    def index(self):return self.data['id']
    def source(self):return Index(self.data['source'])
    def destination(self):return Index(self.data['target'])
    def bondOrder(self):return self.data['order']
    def bondStereo(self):return self.data['stereo']


def collect_paths(root):
    found=[]
    def walk(node):
        if node.tag==f'{{{SVG}}}defs':return
        if node.tag==f'{{{SVG}}}path':found.append(node)
        for child in node:walk(child)
    walk(root);return found


def transform_points(points,m):
    pts=np.asarray(points,float);return np.column_stack((pts,np.ones(len(pts))))@m[:2].T


def box_transform(box,m):
    x1,y1,x2,y2=box;p=transform_points([[x1,y1],[x2,y2]],m)
    return np.r_[p.min(0),p.max(0)].tolist()


def boundary_distance(point,box):
    return float(np.linalg.norm(np.maximum(np.maximum(np.array(box[:2])-point,point-np.array(box[2:])),0)))


def place_charge(rng,atom,atoms,occupancy,width,height,preferred):
    x1,y1,x2,y2=atom['bbox'];cx,cy=atom['center']
    order=[preferred]+[p for p in rng.sample(POSITIONS,len(POSITIONS)) if p!=preferred]
    for shrink in [1.,.9,.8,.7,.6]:
        ww,hh=width*shrink,height*shrink
        for position in order:
            for gap in [rng.uniform(.7,2.0),3.,5.,8.,11.]:
                if position=='upper_right':x,y=x2+gap,y1-hh*.45
                elif position=='upper_left':x,y=x1-gap-ww,y1-hh*.45
                elif position=='above':x,y=cx-ww/2,y1-gap-hh
                elif position=='right':x,y=x2+gap,(y1+y2-hh)/2
                else:x,y=x2+gap,y2+gap
                x+=rng.uniform(-.6,.6);y+=rng.uniform(-.6,.6)
                left,top=math.floor(x-.5),math.floor(y-.5);right,bottom=math.ceil(x+ww+.5),math.ceil(y+hh+.5)
                if left<3 or top<3 or right>occupancy.shape[1]-3 or bottom>occupancy.shape[0]-3:continue
                if occupancy[top:bottom,left:right].any():continue
                center=np.array([x+ww/2,y+hh/2]);own=boundary_distance(center,atom['bbox'])
                others=[boundary_distance(center,a['bbox']) for a in atoms if a['id']!=atom['id'] and a['visible_label']]
                if others and own>min(others)+2:continue
                occupancy[top:bottom,left:right]=True
                return [x,y,x+ww,y+hh],position,shrink
    raise ValueError('No collision-free nearby charge position')


def generate(task):
    record,source_root,out,index,rep,seed,settings=task;out=Path(out);source_root=Path(source_root)
    try:
        side=int(settings.get('canvas_size',640));font_pool=settings.get('fonts') or FONTS
        rng=random.Random(seed);ann=json.loads((source_root/record['annotation']).read_text())
        if not ann['charges'] or any(abs(q['value'])!=1 for q in ann['charges']):raise ValueError('Typography replacement requires +/-1 source charges')
        raw_svg=(source_root/record['svg']).read_text();glyphs,paths=svg_geometry(raw_svg)
        old_affine=np.array(ann['svg_to_image_matrix']);inverse=np.linalg.inv(old_affine)
        coords=transform_points([a['center'] for a in ann['atoms']],inverse)
        owned=ann.get('ownership',{}).get('method')=='exclusive_native_dual_color_v1'
        go={j:g['native_atom_id'] for j,g in enumerate(glyphs) if g.get('native_atom_id') is not None} if owned else None
        po={j:p['native_atom_id'] for j,p in enumerate(paths) if p.get('native_atom_id') is not None} if owned else None
        _,old_charges,excluded=charge_and_atom_glyphs(glyphs,paths,coords,[a['formal_charge'] for a in ann['atoms']],glyph_owners=go,atom_path_owners=po)
        if len(old_charges)!=len(ann['charges']):raise ValueError('Source charge count mismatch')
        tree=ET.fromstring(raw_svg);native_paths=collect_paths(tree)
        if len(native_paths)!=len(paths):raise ValueError('Source SVG path enumeration mismatch')
        parent_map={child:parent for parent in tree.iter() for child in parent}
        for j in excluded:parent_map[native_paths[j]].remove(native_paths[j])
        w,h=ann['width'],ann['height'];occupancy_factor=rng.uniform(.68,.95);scale=(side*.95)*occupancy_factor/max(w,h)
        dx=rng.uniform(side*.025,side*.975-w*scale);dy=rng.uniform(side*.025,side*.975-h*scale)
        scene=np.array([[scale,0,dx],[0,scale,dy],[0,0,1.]])
        full=scene@old_affine
        root=ET.Element(f'{{{SVG}}}svg',{'width':str(side),'height':str(side),'viewBox':f'0 0 {side} {side}','version':'1.1'})
        ET.SubElement(root,f'{{{SVG}}}rect',{'width':str(side),'height':str(side),'fill':'white'})
        definitions=tree.find(f'{{{SVG}}}defs')
        if definitions is not None:root.append(copy.deepcopy(definitions))
        group=ET.SubElement(root,f'{{{SVG}}}g',{'transform':f'matrix({full[0,0]},{full[1,0]},{full[0,1]},{full[1,1]},{full[0,2]},{full[1,2]})'})
        for child in tree:
            if child is definitions:continue
            for node in list(child):
                if node.tag==f'{{{SVG}}}rect':child.remove(node)
            group.append(copy.deepcopy(child))
        body=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ET.tostring(root)))).convert('RGB'))
        occupancy=body.min(2)<245
        new=copy.deepcopy(ann);new['width']=new['height']=side
        for atom in new['atoms']:
            atom['center']=transform_points([atom['center']],scene)[0].tolist();atom['bbox']=box_transform(atom['bbox'],scene)
        for bond in new['bonds']:bond['endpoints']=[new['atoms'][bond['source']]['center'],new['atoms'][bond['target']]['center']]
        new['graph']['coords']=[a['center'] for a in new['atoms']]
        font_file=font_pool[(index+rep)%len(font_pool)];draw=rng.random();tier=settings.get('size_tier') or ('tiny' if draw<.1 else 'small' if draw<.25 else 'medium' if draw<.7 else 'large')
        if tier not in ['tiny','small','medium','large']:raise ValueError('Unsupported charge size tier')
        charges=[];placements=[]
        for qi,q in enumerate(ann['charges']):
            positive=q['value']>0
            if tier=='tiny':ih=rng.uniform(2.5,4) if positive else rng.uniform(.8,1.3);iw=ih*rng.uniform(.9,1.1) if positive else rng.uniform(4,7);pad=1.
            elif tier=='small':ih=rng.uniform(4.5,7) if positive else rng.uniform(1.4,2.2);iw=ih*rng.uniform(.9,1.1) if positive else rng.uniform(7,11);pad=1.25
            elif tier=='medium':ih=rng.uniform(8,12) if positive else rng.uniform(2.8,4.3);iw=ih*rng.uniform(.9,1.1) if positive else rng.uniform(12,19);pad=1.5
            else:ih=rng.uniform(13,18) if positive else rng.uniform(4.5,6);iw=ih*rng.uniform(.9,1.1) if positive else rng.uniform(18,25);pad=2.
            iw*=side/640;ih*=side/640;pad*=side/640
            d,bounds,char=font_glyph(font_file,q['value']);bx,by,bxx,byy=bounds
            preferred=POSITIONS[(index+2*rep+qi)%len(POSITIONS)]
            inkbox,position,shrink=place_charge(rng,new['atoms'][q['atom_id']],new['atoms'],occupancy,iw,ih,preferred)
            x1,y1,x2,y2=inkbox;sx=(x2-x1)/(bxx-bx);sy=(y2-y1)/(byy-by)
            original_path=paths[sorted(excluded,key=lambda j:np.linalg.norm((paths[j]['bbox'][:2]+paths[j]['bbox'][2:])/2-coords[q['atom_id']]))[0]]
            fill=original_path['style'].get('fill','black')
            ET.SubElement(root,f'{{{SVG}}}path',{'id':f'charge-{qi}','data-role':'charge','data-atom-id':str(q['atom_id']),'d':d,'fill':fill,'transform':f'matrix({sx},0,0,{-sy},{x1-bx*sx},{y1+byy*sy})'})
            box=[max(0,x1-pad),max(0,y1-pad),min(side-1,x2+pad),min(side-1,y2+pad)]
            charges.append({**q,'bbox':box,'ink_bbox':inkbox,'bbox_kind':'font_outline_charge_glyph','font_file':font_file,'glyph':char,'position':position,'size_tier':tier,'size_shrink':shrink})
            placements.append({'atom_id':q['atom_id'],'value':q['value'],'font':Path(font_file).name,'position':position,'preferred_position':preferred,'ink_height':y2-y1,'box_height':box[3]-box[1],'ink_width':x2-x1,'tier':tier})
        svg=ET.tostring(root,encoding='unicode');rendered=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGB'))
        if all(b.get('primitive_assignment')=='native_exclusive_highlight' for b in ann['bands']):
            _,final_paths=svg_geometry(svg)
            native={j:p['native_bond_id'] for j,p in enumerate(final_paths) if p.get('native_bond_id') is not None}
            bands,masks,unassigned=make_bands(final_paths,np.array([a['center'] for a in new['atoms']]),[Bond(b) for b in ann['bonds']],set(),np.eye(3),(side,side),path_owners=native)
        else:bands,masks,unassigned=make_bands(paths,coords,[Bond(b) for b in ann['bonds']],excluded,full,(side,side))
        # Parent atom masks use a different affine. Re-rasterize glyph ownership
        # from the transformed SVG; never keep stale cropped mask coordinates.
        if new.get('ownership',{}).get('method')=='exclusive_native_dual_color_v1':
            from .ownership import glyph_mask
            final_glyphs,_=svg_geometry(svg)
            for atom in new['atoms']:
                owned=[j for j,g in enumerate(final_glyphs) if g.get('native_atom_id')==atom['id']]
                if not owned:continue
                ink,bb=glyph_mask(svg,owned,np.eye(3),(side,side))
                if bb is None:raise ValueError('Transformed atom ink is empty')
                atom['bbox']=[max(0,bb[0]-1),max(0,bb[1]-1),min(side-1,bb[2]+1),min(side-1,bb[3]+1)]
                key='atom_ink_'+str(atom['id']);masks[key]=ink
                atom.update(mask_key=key,mask_bbox=bb,ink_area=int(ink.sum()),svg_glyph_ids=[final_glyphs[j]['sequence'] for j in owned])
        new['charges']=charges;new['bands']=bands
        for b,band in zip(new['bonds'],bands):
            b['quad']=band['quad'];b['region_polygon']=band['region_polygon'];pts=np.array(band['quad']);b['bbox']=np.r_[pts.min(0),pts.max(0)].tolist()
        stem=f'{index:05d}_{rep:03d}_'+asset_stem(record['dataset'],record['image_id'],str(seed))
        relative={key:f'assets/charge/{folder}/{stem}.{ext}' for key,folder,ext in [('image','images','png'),('annotation','annotations','json'),('band_target','targets','json'),('masks','masks','npz'),('svg','svg','svg'),('view_check','view_check','jpg'),('overlay','overlays','png')]}
        relative.update(extend_paths(relative))
        new.update({'sample_id':'charge/'+stem,'image_id':stem,'source_training_sample_id':record['sample_id'],'source_training_group':record['group'],'image':relative['image'],'masks':relative['masks'],'svg':relative['svg'],'band_target':relative['band_target'],'svg_to_image_matrix':np.eye(3).tolist(),'generator_revision':5,'charge_augmentation':{'recipe':'font_outline_replacement_with_collision_free_layout_v1','seed':seed,'parent_sample_id':record['sample_id'],'parent_group':record['group'],'font_file':font_file,'size_tier':tier,'source_to_image_matrix':scene.tolist(),'charge_placements':placements,'chemical_graph_changed':False,'training_only':True}})
        new['style']['charge_augmentation']=new['charge_augmentation']
        new['charge_augmentation']['font_file']=Path(font_file).name
        for q in new['charges']:q['font_file']=Path(q['font_file']).name
        attach_lines(new)
        for path in relative.values():(out/path).parent.mkdir(parents=True,exist_ok=True)
        marked=overlay(rendered,new,masks)
        Image.fromarray(rendered).save(out/relative['image']);marked.save(out/relative['overlay'])
        source_image=np.array(Image.open(source_root/record['image']).convert('RGB'))
        review(source_image,rendered,marked,f'TRAIN charge augmentation | {stem} | {Path(font_file).name} | {tier}').save(out/relative['view_check'],quality=90)
        write_line_artifacts(out,new,relative)
        (out/relative['svg']).write_text(svg);(out/relative['annotation']).write_text(json.dumps(new,ensure_ascii=False,separators=(',',':'))+'\n');np.savez_compressed(out/relative['masks'],**masks)
        target={'schema':'independent_band_quad_v1','image':relative['image'],'size':[side,side],'labels':[b['category_id']-1 for b in bands],'quads':[b['quad'] for b in bands]};(out/relative['band_target']).write_text(json.dumps(target,separators=(',',':'))+'\n')
        result={**record,**relative,'sample_id':'charge/'+stem,'parent_id':record['sample_id'],'image_id':stem,'variant':'charge_augmented','has_charge':True,'status':'ok','placements':placements,'source_parent_id':record['sample_id']}
        return result
    except Exception as exc:return {'status':'error','source_parent_id':record['sample_id'],'index':index,'rep':rep,'error':str(exc)}
