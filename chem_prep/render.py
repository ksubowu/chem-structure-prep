"""CSV-independent deterministic molecular rendering and annotation extraction."""
import io,math,random,xml.etree.ElementTree as ET
from pathlib import Path
import cairosvg,cv2,numpy as np
from PIL import Image
from .backend import create_backend,set_font
from .chemistry import load_molecule
from .augmentation import prepare,assign_dummy_glyphs
from .svg import svg_geometry,charge_and_atom_glyphs,make_bands
from .visuals import image_stats,overlay,review
from .io import read_json,write_json
from .ownership import infer_ownership,glyph_mask,annotate_svg
from .lines import attach as attach_lines,write_artifacts as write_line_artifacts
BOND_NAMES={1:'single',2:'double',3:'triple',4:'aromatic',5:'solid wedge',6:'dashed wedge'}


def stable_molfile(mol):
    lines=mol.molfile().splitlines()
    if len(lines)>1:lines[1]='  ChemPrep          2D'
    return '\n'.join(lines)+'\n'


def _render_one(job):
    root=Path(job['root']);paths=job['paths'];settings=job['render'];source=job['source']
    try:
        cv2.setNumThreads(1);reference=None;size=settings['canvas_size'];w,h=(size,size) if isinstance(size,int) else size
        if source.get('reference'):
            ref=Image.open(source['reference']).convert('RGB');ref.thumbnail((w,h),Image.Resampling.LANCZOS)
            ref_canvas=Image.new('RGB',(w,h),'white');ref_canvas.paste(ref,((w-ref.width)//2,(h-ref.height)//2));reference=np.array(ref_canvas)
        padding=settings.get('padding',24)
        if reference is None:obox=[padding,padding,w-padding,h-padding];ostroke=None
        else:obox,ostroke=image_stats(reference)
        ig,renderer=create_backend();ig.setOption('render-background-color','1,1,1');ig.setOption('render-stereo-style','none');ig.setOption('render-label-mode','hetero')
        font=random.Random(job['seed']).choice(settings.get('font_families',['DejaVu Sans','DejaVu Serif']))
        engine_font=set_font(ig,font);ig.setOption('render-implicit-hydrogens-visible',True);ig.setOption('render-bond-length',float(settings.get('bond_length',45)));ig.setOption('render-margins',8,8)
        if source.get('molfile'):
            mol=ig.loadMolecule(source['molfile']);replaced=[]
        else:mol,replaced=load_molecule(ig,source['smiles'])
        if not 0<mol.countAtoms()<=settings.get('max_atoms',500):raise ValueError('Molecule atom count outside configured range')
        aug=None;color=None
        if job['is_color']:mol,aug,color=prepare(ig,mol,{'seed':job['seed'],'probability':job['dummy_probability'],'labels':job['labels']})
        if not settings.get('preserve_source_coordinates',False):mol.layout()
        relative=float(settings.get('relative_thickness',1.0))
        if reference is not None and settings.get('match_reference',True):
            ig.setOption('render-output-format','png');trial=cv2.imdecode(np.frombuffer(bytes(renderer.renderToBuffer(mol)),np.uint8),cv2.IMREAD_COLOR)
            rb,stroke=image_stats(cv2.cvtColor(trial,cv2.COLOR_BGR2RGB));scale=min((obox[2]-obox[0])/max(1,rb[2]-rb[0]),(obox[3]-obox[1])/max(1,rb[3]-rb[1]))
            relative=float(np.clip(ostroke/max(.1,scale*stroke),.5,2))
        ig.setOption('render-relative-thickness',relative);ig.setOption('render-output-format','svg');svg=bytes(renderer.renderToBuffer(mol)).decode()
        if settings.get('reject_non_single_wedges',False) and any(b.bondOrder()!=1 and b.bondStereo() in (5,6) for b in mol.iterateBonds()):
            raise ValueError('Non-single bond carries wedge stereo; reject this depiction')
        coords=np.array([a.coords() for a in mol.iterateAtoms()],float);vb=ET.fromstring(svg).get('viewBox').split();rw,rh=int(float(vb[2])),int(float(vb[3]))
        if max(rw,rh)>settings.get('max_svg_side',20000):raise ValueError('Rendered SVG too large')
        raster=cairosvg.svg2png(bytestring=svg.encode(),output_width=rw,output_height=rh);rgb=np.array(Image.open(io.BytesIO(raster)).convert('RGB'))
        glyph_owners,path_owners,ownership=infer_ownership(ig,renderer,mol,svg)
        atom_path_owners={j:hit['atom_ids'][0] for hit in ownership['atoms'] for j in hit['path_indices']}
        svg=annotate_svg(svg,glyph_owners,path_owners,atom_path_owners)
        glyphs,svg_paths=svg_geometry(svg);formal=[a.charge() or 0 for a in mol.iterateAtoms()];grouped,raw_charges,exclude=charge_and_atom_glyphs(glyphs,svg_paths,coords,formal,glyph_owners=glyph_owners,atom_path_owners=atom_path_owners)
        if aug and aug['applied']:
            aug['label_glyph_count']=sum(owner==aug['dummy_atom_id'] for owner in glyph_owners.values())
            aug['label_box_source']='exclusive_native_dual_color_v1'
        rb,_=image_stats(rgb)
        rb=[max(0,min(rb[0],math.floor(coords[:,0].min())-2)),max(0,min(rb[1],math.floor(coords[:,1].min())-2)),min(rw,max(rb[2],math.ceil(coords[:,0].max())+3)),min(rh,max(rb[3],math.ceil(coords[:,1].max())+3))]
        scale=min((obox[2]-obox[0])/max(1,rb[2]-rb[0]),(obox[3]-obox[1])/max(1,rb[3]-rb[1]));nw=max(1,round((rb[2]-rb[0])*scale));nh=max(1,round((rb[3]-rb[1])*scale))
        sx=nw/(rb[2]-rb[0]);sy=nh/(rb[3]-rb[1]);left=max(0,min(w-nw,round((obox[0]+obox[2]-nw)/2)));top=max(0,min(h-nh,round((obox[1]+obox[3]-nh)/2)))
        affine=np.array([[sx,0,left-rb[0]*sx],[0,sy,top-rb[1]*sy],[0,0,1.]])
        rendered=np.full((h,w,3),255,np.uint8);rendered[top:top+nh,left:left+nw]=cv2.resize(rgb[rb[1]:rb[3],rb[0]:rb[2]],(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        def point(p):return [round((float(p[0])-rb[0])*sx+left,4),round((float(p[1])-rb[1])*sy+top,4)]
        def box(b,pad=1):
            a=point(b[:2]);z=point(b[2:]);return [max(0,a[0]-pad),max(0,a[1]-pad),min(w-1,z[0]+pad),min(h-1,z[1]+pad)]
        atoms=[]
        for i,a in enumerate(mol.iterateAtoms()):
            p=point(coords[i]);visible=bool(grouped[i]);radius=max(2,45*scale*.08)
            if visible:
                g=np.array(grouped[i]);bb=box([g[:,0].min(),g[:,1].min(),g[:,2].max(),g[:,3].max()])
            else:bb=[max(0,p[0]-radius),max(0,p[1]-radius),min(w-1,p[0]+radius),min(h-1,p[1]+radius)]
            atoms.append({'id':i,'symbol':a.symbol(),'formal_charge':formal[i],'center':p,'bbox':bb,'visible_label':visible,'is_pseudoatom':a.isPseudoatom(),'bbox_kind':'native_highlight_owned_glyph_union' if visible else 'unlabeled_vertex_proxy'})
        n=len(atoms);edges=[[0]*n for _ in range(n)];bonds=[]
        for b in mol.iterateBonds():
            s,t=b.source().index(),b.destination().index();order,stereo=b.bondOrder(),b.bondStereo();code=stereo if stereo in (5,6) else order
            if code not in BOND_NAMES:raise ValueError(f'Unsupported chemical bond class: {code}')
            edges[s][t]=code;edges[t][s]=11-code if code in (5,6) else code
            bonds.append({'id':b.index(),'source':s,'target':t,'order':order,'stereo':stereo,'type':BOND_NAMES[code],'endpoints':[atoms[s]['center'],atoms[t]['center']]})
        charges=[{'id':i,'atom_id':q['atom_id'],'value':q['value'],'label':q['label'],'bbox':box(q['raw_bbox'],.75),'bbox_kind':'rendered_charge_sign_and_magnitude'} for i,q in enumerate(raw_charges)]
        bands,masks,unassigned=make_bands(svg_paths,coords,list(mol.iterateBonds()),exclude,affine,(w,h),path_owners=path_owners)
        # Save actual atom-label ink masks. Bare vertices have anchor boxes,
        # not invented character ink. Charge magnitudes remain independent.
        for atom in atoms:
            owned=[j for j,owner in glyph_owners.items() if owner==atom['id'] and any(np.array_equal(glyphs[j]['bbox'],bb) for bb in grouped[owner])]
            if not owned:continue
            ink,bb=glyph_mask(svg,owned,affine,(w,h))
            if bb is None:raise ValueError('Owned atom glyphs rasterized to empty ink')
            atom['bbox']=[max(0,bb[0]-1),max(0,bb[1]-1),min(w-1,bb[2]+1),min(h-1,bb[3]+1)]
            atom['bbox_kind']='native_highlight_label_ink_bounds'
            key='atom_ink_'+str(atom['id']);masks[key]=ink
            atom.update({'mask_key':key,'mask_bbox':bb,'ink_area':int(ink.sum()),'svg_glyph_ids':[glyphs[j]['sequence'] for j in owned]})
        for b,band in zip(bonds,bands):
            pts=np.array(band['quad']);b.update({'band_id':band['id'],'quad':band['quad'],'region_polygon':band['region_polygon'],'bbox':np.r_[pts.min(0),pts.max(0)].tolist(),'bbox_kind':'actual_band_axis_aligned_envelope'})
        if aug and aug['applied']:atoms[aug['dummy_atom_id']].update({'is_added_dummy':True,'bbox_kind':'owned_dummy_label_native_ink_bounds'})
        ann={'schema_version':2,'generator_revision':6,'sample_id':job['sample_id'],'dataset':source['dataset'],'image_id':source['id'],'source_SMILES':source['smiles'],'SMILES':mol.smiles(),'SMILES_format':'Indigo_CXSMILES_with_pseudoatom_labels','rendered_molfile':stable_molfile(mol),'width':w,'height':h,'bbox_format':'xyxy_pixels','coordinate_system':'top-left origin, x right, y down; rendered-image pixels','image':paths['image'],'masks':paths['masks'],'svg':paths['svg'],'band_target':paths['band_target'],'svg_to_image_matrix':affine.tolist(),'atoms':atoms,'charges':charges,'bonds':bonds,'bands':bands,'graph':{'coords':[a['center'] for a in atoms],'symbols':[a['symbol'] for a in atoms],'edges':edges,'num_atoms':n},'ownership':ownership,'augmentation':aug,'style':{'font_family':font,'engine_font_family':engine_font,'relative_thickness':relative,'reference_matched':reference is not None,'color':color},'pseudoatom_labels_preserved':replaced,'mask_storage':'cropped nonexclusive bool arrays; paste at each band.mask_bbox (xyxy)','svg_unassigned_absolute_path_count':len(unassigned),'limitations':['Atom labels and bond primitives are owned by geometry-checked native dual-color highlighting.','Unlabeled atom boxes are geometric anchor proxies.']}
        attach_lines(ann)
        for p in paths.values():(root/p).parent.mkdir(parents=True,exist_ok=True)
        marked=overlay(rendered,ann,masks);Image.fromarray(rendered).save(root/paths['image']);marked.save(root/paths['overlay']);review(reference,rendered,marked,job['sample_id']).save(root/paths['view_check'],quality=90)
        write_line_artifacts(root,ann,paths)
        (root/paths['svg']).write_text(svg);np.savez_compressed(root/paths['masks'],**masks);write_json(root/paths['annotation'],ann)
        write_json(root/paths['band_target'],{'schema':'independent_band_quad_v1','image':paths['image'],'size':[w,h],'labels':[b['category_id']-1 for b in bands],'quads':[b['quad'] for b in bands]})
        return {'sample_id':job['sample_id'],'dataset':source['dataset'],'image_id':source['id'],'variant':job['variant'],'parent_id':job.get('parent_id'),'source_SMILES':source['smiles'],'SMILES':ann['SMILES'],'is_color':job['is_color'],'dummy_applied':bool(aug and aug['applied']),'dummy_label':aug.get('label','') if aug else '',**paths,'has_charge':bool(charges),'charge_values':[q['value'] for q in charges],'status':'ok','job_fingerprint':job['fingerprint']}
    except Exception as exc:return {'sample_id':job['sample_id'],'dataset':source['dataset'],'image_id':source['id'],'status':'error','error':f'{type(exc).__name__}: {exc}'}


def render_one(job):
    # One atomic per-sample checkpoint survives interruption/native worker loss.
    import hashlib
    result=_render_one(job)
    key=hashlib.sha256(job['sample_id'].encode()).hexdigest()
    write_json(Path(job['root'])/'_render_checkpoints'/(key+'.json'),result)
    return result
