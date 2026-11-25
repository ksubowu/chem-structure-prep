"""Portable manifests, deterministic identities, hashing and path helpers."""
import csv,hashlib,json,os,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['sample_id']
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def write_manifest(root,name,rows):
    write_json(Path(root)/(name+'.json'),rows);write_csv(Path(root)/(name+'.csv'),rows)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def file_hashes(root,rows,files=()):
    paths=set(files)
    for r in rows:
        for key in ['image','annotation','band_target','masks','svg','view_check','overlay','line_target','line_overlay','band_line_view_check']:
            if key not in r:continue
            if r.get(key):paths.add(r[key])
    ordered=sorted(paths)
    with ThreadPoolExecutor(max_workers=12) as pool:values=list(pool.map(lambda p:digest(Path(root)/p),ordered))
    return dict(zip(ordered,values))
def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(json.dumps([seed,*parts],ensure_ascii=False,sort_keys=True).encode()).digest()[:8],'big')
def slug(text):return re.sub(r'[^A-Za-z0-9_.-]+','_',str(text)).strip('._')[:72] or 'sample'
def asset_stem(dataset,ident,smiles):return slug(dataset)+'_'+slug(ident)+'_'+hashlib.sha256(json.dumps([str(dataset),str(ident),smiles],ensure_ascii=False).encode()).hexdigest()[:12]
