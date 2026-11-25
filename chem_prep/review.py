"""Self-contained local review page and compact preview images."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
from .io import read_json


def build(root):
    root=Path(root);rows=read_json(root/'data_manifest.json');data=[]
    flags={r['sample_id']:r['band_ids'] for r in read_json(root/'validation_report.json').get('review_flags',[])} if (root/'validation_report.json').exists() else {}
    picks={}
    for r in rows:
        data.append({'id':r['sample_id'],'dataset':r['dataset'],'variant':r['variant'],'charged':r['has_charge'],'dummy':r['dummy_applied'],'view':r.get('band_line_view_check',r['view_check']),'ann':r['annotation'],'band_ids':flags.get(r['sample_id'],[])})
        picks.setdefault((r['dataset'],r['variant'],r['has_charge']),r)
    selected=list(picks.values())[:12]
    images=[]
    for r in selected:
        with Image.open(root/r.get('band_line_view_check',r['view_check'])) as im:images.append(im.convert('RGB'))
    if images:
        width=max(i.width for i in images);height=sum(i.height for i in images);preview=Image.new('RGB',(width,height),'white');y=0
        for im in images:preview.paste(im,(0,y));y+=im.height
        preview.save(root/'preview.jpg',quality=88)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Chemical structure preparation review</title><style>body{font:16px sans-serif;margin:24px;background:#eee}article{background:white;padding:12px;margin:16px 0}img{max-width:100%;height:auto}input,select,button{font:inherit;padding:6px;margin:5px}</style><h1>化学结构数据检查</h1><p>蓝框：原子标签；粉框：独立电荷；绿框：隐式原子顶点；彩色区域：实际键墨迹与 band。所有坐标属于渲染图。</p><input id="q" placeholder="sample ID / dataset"><label><input type="checkbox" id="charged">带电荷</label><label><input type="checkbox" id="dummy">dummy 标签</label><label><input type="checkbox" id="flagged">待复核 bands</label><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span><main id="list"></main><script>const data=DATA;let page=0;const size=16,q=document.getElementById('q'),c=document.getElementById('charged'),d=document.getElementById('dummy'),f=document.getElementById('flagged');function render(){const r=data.filter(x=>(x.id.includes(q.value)||x.dataset.includes(q.value))&&(!c.checked||x.charged)&&(!d.checked||x.dummy)&&(!f.checked||x.band_ids.length));page=Math.max(0,Math.min(page,Math.ceil(r.length/size)-1));const l=document.getElementById('list');l.replaceChildren();for(const x of r.slice(page*size,(page+1)*size)){const a=document.createElement('article'),p=document.createElement('p');p.textContent=x.id+' | '+x.variant+(x.band_ids.length?' | Review bands: '+x.band_ids.join(', '):'');a.append(p);const link=document.createElement('a');link.href=x.ann;link.textContent='annotation JSON';a.append(link);const im=document.createElement('img');im.src=x.view;im.loading='lazy';a.append(im);l.append(a)}document.getElementById('count').textContent=r.length+' images; page '+(page+1)}q.oninput=c.onchange=d.onchange=f.onchange=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script></html>'''
    (root/'index.html').write_text(page.replace('DATA',json.dumps(data,ensure_ascii=True).replace('<','\\u003c')))
    return {'samples':len(rows),'preview_cases':len(selected)}
