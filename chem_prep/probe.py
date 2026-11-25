"""Export inspectable exclusive atom/group/bond highlight probes and pixel masks."""
import io,json,xml.etree.ElementTree as ET
from pathlib import Path
import cairosvg,numpy as np
from PIL import Image,ImageDraw
from .backend import create_backend,set_font
from .chemistry import load_molecule
from .ownership import OwnershipProbe
from .io import write_json


def raster(svg,affine,size):
    tree=ET.fromstring(svg);w,h=size;root=ET.Element('{http://www.w3.org/2000/svg}svg',{'width':str(w),'height':str(h),'viewBox':f'0 0 {w} {h}'})
    ET.SubElement(root,'{http://www.w3.org/2000/svg}rect',{'width':str(w),'height':str(h),'fill':'white'})
    m=affine;g=ET.SubElement(root,'{http://www.w3.org/2000/svg}g',{'transform':f'matrix({m[0,0]},{m[1,0]},{m[0,1]},{m[1,1]},{m[0,2]},{m[1,2]})'})
    # Convert nested svg to g; no nested viewport clipping/automatic resize.
    tree.tag='{http://www.w3.org/2000/svg}g'
    for k in ['width','height','viewBox']:tree.attrib.pop(k,None)
    g.append(tree)
    return np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ET.tostring(root)))).convert('RGB'))


def export(smiles,output,atom_ids=None,bond_ids=None,group_smarts=None,match_index=0,all_atoms=False,all_bonds=False,colored=False,font='DejaVu Sans',size=640,include_internal_bonds=False):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    ig,renderer=create_backend();set_font(ig,font)
    for k,v in [('render-output-format','svg'),('render-label-mode','hetero'),('render-background-color','1,1,1'),('render-bond-length',45),('render-implicit-hydrogens-visible',True),('render-highlight-thickness-enabled',False),('render-highlighted-labels-visible',False),('render-coloring',colored)]:ig.setOption(k,v)
    mol,_=load_molecule(ig,smiles);mol.layout();baseline=bytes(renderer.renderToBuffer(mol)).decode();vb=list(map(float,ET.fromstring(baseline).get('viewBox').split()));scale=(size-32)/max(vb[2:]);affine=np.array([[scale,0,(size-scale*vb[2])/2],[0,scale,(size-scale*vb[3])/2],[0,0,1.]])
    selections=[]
    if atom_ids is not None or bond_ids is not None:selections.append(('selection',atom_ids or [],bond_ids or []))
    if all_atoms:selections.extend((f'atom_{a.index()}',[a.index()],[]) for a in mol.iterateAtoms())
    if all_bonds:selections.extend((f'bond_{b.index()}',[],[b.index()]) for b in mol.iterateBonds())
    if group_smarts:
        query=ig.loadSmarts(group_smarts)
        # Native match iterators borrow their matcher; keep it alive explicitly.
        matcher=ig.substructureMatcher(mol);iterator=matcher.iterateMatches(query);selected=None
        if match_index<0:raise ValueError('SMARTS match_index out of range')
        for i,match in enumerate(iterator):
            if i==match_index:
                selected=[match.mapAtom(a).index() for a in query.iterateAtoms()];break
        if selected is None:raise ValueError('SMARTS match_index out of range')
        bs=[b.index() for b in mol.iterateBonds() if b.source().index() in selected and b.destination().index() in selected] if include_internal_bonds else []
        selections.append(('group',selected,bs))
    if not selections:raise ValueError('Select --atom-ids, --bond-ids, --group-smarts, --all-atoms or --all-bonds')
    out.mkdir(parents=True);rgb=raster(baseline,affine,(size,size));Image.fromarray(rgb).save(out/'baseline.png');(out/'baseline.svg').write_text(baseline)
    results=[]
    with OwnershipProbe(ig,renderer,mol,baseline) as probe:
        for name,atoms,bonds in selections:
            hit=probe.select(atoms,bonds,keep_svg=True);images=[raster(s,affine,(size,size)) for s in hit.pop('probe_svgs')]
            mask=np.any(images[0]!=images[1],axis=2);ys,xs=np.nonzero(mask)
            bb=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
            ink_bb=bb;vertex_boxes=[]
            for i in hit['unlabeled_vertex_ids']:
                x,y=(np.r_[mol.getAtom(i).coords(),1]@affine[:2].T).tolist();radius=max(2.,45*scale*.08)
                vertex_boxes.append([max(0,x-radius),max(0,y-radius),min(size-1,x+radius),min(size-1,y+radius)])
            if vertex_boxes:
                boxes=vertex_boxes+([bb] if bb else []);b=np.array(boxes);bb=[float(b[:,0].min()),float(b[:,1].min()),float(b[:,2].max()),float(b[:,3].max())]
            folder=out/name;folder.mkdir();Image.fromarray(images[0]).save(folder/'probe_red.png');Image.fromarray(images[1]).save(folder/'probe_cyan.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png');np.savez_compressed(folder/'mask.npz',mask=mask)
            marked=rgb.copy();marked[mask]=(.4*marked[mask]+.6*np.array([255,0,180])).astype('uint8');im=Image.fromarray(marked)
            if bb:ImageDraw.Draw(im).rectangle(bb,outline='magenta',width=2)
            im.save(folder/'overlay.png');hit.update(name=name,bbox_xyxy=bb,ink_bbox_xyxy=ink_bb,vertex_proxy_boxes=vertex_boxes,ink_area=int(mask.sum()),mask='mask.png');write_json(folder/'selection.json',hit);results.append(hit)
    report={'method':'exclusive_native_dual_color_v1','SMILES':smiles,'width':size,'height':size,'font':font,'colored':colored,'svg_to_image_matrix':affine.tolist(),'configuration_fixed':True,'geometry_verified':True,'selections':results,'notes':['Masks are the RGB difference between exclusive red/cyan probes, not a red-pixel threshold.','Ordinary unlabeled skeletal C has no glyph ink and uses its graph anchor in production annotations.','Group masks include internal bonds only when requested.']}
    write_json(out/'report.json',report)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Exclusive ownership probes</title><h1>Native exclusive highlights</h1><img width="360" src="baseline.png">'+''.join('<h2>'+r['name']+'</h2>'+''.join('<img width="280" src="'+r['name']+'/'+f+'">' for f in ['probe_red.png','probe_cyan.png','mask.png','overlay.png']) for r in results))
    return {'output':str(out),'selections':len(results),'geometry_verified':True}
