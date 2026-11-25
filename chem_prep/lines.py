"""A second, type-independent bond geometry: source-center to target-center."""
import copy,hashlib,html,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .io import read_json,write_json,write_manifest

MODES=('atom_center','box_center')


def attach(ann,endpoint_mode='atom_center'):
    if endpoint_mode not in MODES:raise ValueError('Unsupported line endpoint mode: '+endpoint_mode)
    atoms={a['id']:a for a in ann['atoms']};bands={b['bond_id']:b for b in ann['bands']};lines=[]
    def center(atom):
        if endpoint_mode=='atom_center':return list(atom['center'])
        x1,y1,x2,y2=atom['bbox'];return [(x1+x2)/2,(y1+y2)/2]
    for bond in ann['bonds']:
        source,target=bond['source'],bond['target'];points=[center(atoms[source]),center(atoms[target])]
        if not np.isfinite(points).all():raise ValueError('Nonfinite line endpoints')
        line={'id':bond['id'],'bond_id':bond['id'],'source':source,'target':target,'category_id':bands[bond['id']]['category_id'],'endpoints':points,'endpoint_mode':endpoint_mode}
        lines.append(line);bond['line_id']=line['id'];bond['line']=copy.deepcopy(points)
    ann['lines']=lines;ann['line_endpoint_mode']=endpoint_mode
    return lines


def target(ann):
    return {'schema':'independent_bond_line_v1','image':ann['image'],'size':[ann['width'],ann['height']],'endpoint_mode':ann['line_endpoint_mode'],'labels':[v['category_id']-1 for v in ann['lines']],'lines':[v['endpoints'] for v in ann['lines']]}


def extend_paths(paths):
    # Keep variant/dataset/stem, changing just the artifact directory/suffix.
    image=Path(paths['image']);base=image.parent.parent;stem=image.stem
    return {'line_target':str(base/'line_targets'/(stem+'.json')),'line_overlay':str(base/'line_overlays'/(stem+'.png')),'band_line_view_check':str(base/'band_line_view_check'/(stem+'.jpg'))}


def draw(image,ann):
    im=image.copy().convert('RGB');d=ImageDraw.Draw(im);width=max(1,round(min(im.size)/300));radius=max(2,width+1)
    for line in ann['lines']:
        # Same identity hue as the band overlay, independent of chemical type.
        import cv2
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(line['id']*137.5)%180,210,210]]]),cv2.COLOR_HSV2RGB)[0,0])
        p,q=map(tuple,line['endpoints']);d.line([p,q],fill=color,width=width)
        for x,y in [p,q]:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
        x,y=(np.array(p)+np.array(q))/2;d.text((x+2,y+2),'L'+str(line['id']),fill=color,stroke_width=1,stroke_fill='white')
    return im


def write_artifacts(root,ann,paths):
    root=Path(root);ann['line_target']=paths['line_target']
    for field in ['line_target','line_overlay','band_line_view_check']:(root/paths[field]).parent.mkdir(parents=True,exist_ok=True)
    write_json(root/paths['line_target'],target(ann));image=Image.open(root/paths['image']).convert('RGB');marked=draw(image,ann);marked.save(root/paths['line_overlay'])
    band=Image.open(root/paths['overlay']).convert('RGB');canvas=Image.new('RGB',(1800,640),'white');d=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate([(image,'Rendered image'),(band,'Bond bands + atom / charge boxes'),(marked,'Bond lines: atom centers; L# = bond ID' if ann['line_endpoint_mode']=='atom_center' else 'Bond lines: box centers; L# = bond ID')]):
        im=im.copy();im.thumbnail((580,570));canvas.paste(im,(600*i+(600-im.width)//2,50+(570-im.height)//2));d.text((600*i+12,14),title,fill='black')
    canvas.save(root/paths['band_line_view_check'],quality=92)


def add_dataset(dataset,output,endpoint_mode='atom_center'):
    source=Path(dataset).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Source and output must be separate directories')
    if (source/'val.json').exists():raise ValueError('add-lines expects an unsplit dataset; frozen splits are supported by new run/render generation, not this migration command')
    rows=read_json(source/'data_manifest.json')
    for row in rows:
        for key in ['image','annotation','masks','svg','band_target','overlay','view_check']:
            p=Path(row[key])
            if p.is_absolute() or not (source/p).resolve().is_relative_to(source):raise ValueError('add-lines requires portable relative artifact paths: '+key)
    shutil.copytree(source,out);checks=[];counts=Counter()
    for row in rows:
        before=read_json(source/row['annotation']);ann=copy.deepcopy(before);lines=attach(ann,endpoint_mode);paths=extend_paths(row);row.update(paths)
        write_artifacts(out,ann,row);write_json(out/row['annotation'],ann)
        for field in ['image','svg','masks','band_target']:
            old=hashlib.sha256((source/row[field]).read_bytes()).hexdigest();new=hashlib.sha256((out/row[field]).read_bytes()).hexdigest()
            if old!=new:raise ValueError('Preserved artifact changed: '+field)
        if before['atoms']!=ann['atoms'] or before['bands']!=ann['bands']:raise ValueError('Existing atom/band annotation changed')
        checks.append({'sample_id':row['sample_id'],'bonds':len(lines),'image_svg_masks_band_target_unchanged':True});counts.update(v['category_id'] for v in lines)
    write_manifest(out,'data_manifest',rows)
    if (out/'render_manifest.json').exists():
        prior=read_json(out/'render_manifest.json');lookup={r['sample_id']:r for r in rows}
        write_manifest(out,'render_manifest',[lookup.get(r['sample_id'],r) for r in prior])
    report={'source_dataset':str(source),'images':len(rows),'black_white':sum(not r['is_color'] for r in rows),'color':sum(r['is_color'] for r in rows),'bonds':sum(r['bonds'] for r in checks),'line_endpoint_mode':endpoint_mode,'category_counts':dict(counts),'images_svg_masks_and_existing_band_targets_unchanged':True,'atom_and_band_arrays_unchanged':True,'original_dataset_modified':False,'checks':checks}
    write_json(out/'line_extension_report.json',report)
    from .review import build
    build(out)
    (out/'README_LINES.txt').write_text('每条化学键同时有 band 和 line 两种几何标注。\n默认 line 端点取 source / target 的 atoms[].center，顺序不随键型改变。\n双键、三键、芳香键、实楔键和虚楔键同样只各有一条 center-to-center line，键型标签照常保留。\nannotations[].lines 与 bonds[].line / line_id 为关联形式；line_targets/ 为独立的 labels + lines 监督格式。\nline_overlays/ 与 band_line_view_check/ 提供叠加及并排核查。\n原始图片、SVG、mask、band target、atom / band 数组逐项核对保持不变。原数据集未修改。\n输出为独立扩展副本，没有修改已有训练/验证划分。\n')
    return {k:v for k,v in report.items() if k!='checks'}
