"""Configuration schema and CSV ingestion. Paths are relative to the config file."""
import copy,csv,glob,os
from pathlib import Path
import yaml
from .io import digest

DEFAULTS={
 'seed':20261113,'workers':4,'exclude_csv':[],
 'render':{'canvas_size':640,'padding':24,'font_families':['DejaVu Sans','DejaVu Serif'],'bond_length':45,'relative_thickness':1.0,'match_reference':True,'max_atoms':500,'max_svg_side':20000},
 'color':{'count':0,'dummy_probability':.5,'labels_file':None},
 'split':{'validation_fraction':.1,'seed':20261113,'stratify':['dataset','variant','dummy_applied','has_charge'],'solver_seconds':30},
 'charge_augmentation':{'count':0,'seed':20261114,'canvas_size':640,'fonts':None},
 'balance':{'enabled':False,'charged_fraction':.25,'seed':20261114},
 'quality':{'fail_on_render_error':False},'review':{'enabled':True}}
ALLOWED=set(DEFAULTS)|{'inputs','output'}


def merge(default,override):
    result=copy.deepcopy(default)
    for key,value in override.items():
        if key not in result:raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(value,dict) and isinstance(result[key],dict):result[key]=merge(result[key],value)
        else:result[key]=value
    return result


def resolve(path,base):
    value=os.path.expandvars(str(path))
    if '${' in value:raise ValueError(f'Unresolved environment variable in path: {path}')
    p=Path(value).expanduser();return str((p if p.is_absolute() else Path(base)/p).resolve())


def load(path):
    path=Path(path).resolve();raw=yaml.safe_load(path.read_text())
    if not isinstance(raw,dict):raise ValueError('Config must be a mapping')
    unknown=set(raw)-ALLOWED
    if unknown:raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    cfg=merge(DEFAULTS,{k:v for k,v in raw.items() if k in DEFAULTS})
    if not raw.get('inputs'):raise ValueError('inputs must contain at least one CSV specification')
    cfg['inputs']=[]
    for entry in raw['inputs']:
        if not isinstance(entry,dict) or 'csv' not in entry:raise ValueError('Each input requires csv')
        unknown=set(entry)-{'name','csv','smiles_column','id_column','image_column','image_root','limit','font_families'}
        if unknown:raise ValueError(f'Unknown input keys: {unknown}')
        item=dict(entry);item['csv']=resolve(item['csv'],path.parent)
        if item.get('image_root'):item['image_root']=resolve(item['image_root'],path.parent)
        cfg['inputs'].append(item)
    cfg['output']=resolve(raw.get('output','output'),path.parent)
    if cfg['color']['labels_file']:cfg['color']['labels_file']=resolve(cfg['color']['labels_file'],path.parent)
    if cfg['charge_augmentation']['fonts']:cfg['charge_augmentation']['fonts']=[resolve(p,path.parent) for p in cfg['charge_augmentation']['fonts']]
    cfg['config_file']=str(path)
    if not 0<cfg['split']['validation_fraction']<1:raise ValueError('validation_fraction must be between 0 and 1')
    if not 0<=cfg['color']['dummy_probability']<=1:raise ValueError('dummy_probability must be in [0,1]')
    if not 0<cfg['balance']['charged_fraction']<1:raise ValueError('charged_fraction must be in (0,1)')
    if cfg['balance']['enabled'] and cfg['color']['count']<0:raise ValueError('color.count must be nonnegative')
    if cfg['workers']<1:raise ValueError('workers must be positive')
    size=cfg['render']['canvas_size'];dims=[size,size] if isinstance(size,int) else size
    if len(dims)!=2 or min(dims)<64 or cfg['render']['padding']>=min(dims)/2:raise ValueError('Invalid canvas_size / padding')
    if cfg['charge_augmentation']['canvas_size']<64:raise ValueError('charge augmentation canvas_size must be >=64')
    for section in ['color','charge_augmentation']:
        if cfg[section]['count']<0:raise ValueError(section+'.count must be nonnegative')
    return cfg


def ingest(cfg):
    records=[];issues=[];files=[];seen=set();names=set()
    for spec in cfg['inputs']:
        matches=sorted(glob.glob(spec['csv']))
        if not matches:raise FileNotFoundError(spec['csv'])
        for filename in matches:
            path=Path(filename)
            if path.name.casefold() in {s.casefold() for s in cfg['exclude_csv']}:continue
            dataset=spec.get('name',path.stem)
            if len(matches)>1 and spec.get('name'):dataset+='_'+path.stem
            if dataset in names:raise ValueError(f'Dataset name collision: {dataset}')
            names.add(dataset);files.append({'name':dataset,'file':str(path),'sha256':digest(path)})
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);smcol=spec.get('smiles_column','SMILES');idcol=spec.get('id_column','image_id');imcol=spec.get('image_column')
                if smcol not in (reader.fieldnames or []):raise ValueError(f'{path}: missing SMILES column {smcol}')
                for index,row in enumerate(reader,1):
                    if spec.get('limit') and index>spec['limit']:break
                    ident=str(row.get(idcol) or index);smiles=(row.get(smcol) or '').strip();key=(dataset,ident)
                    reason='empty SMILES' if not smiles else 'duplicate dataset/id' if key in seen else None
                    seen.add(key)
                    if reason:issues.append({'dataset':dataset,'id':ident,'row':index,'error':reason});continue
                    image=(row.get(imcol) or '').strip() if imcol else ''
                    ref=resolve(image,spec.get('image_root',path.parent)) if image else None
                    if ref and not Path(ref).is_file():issues.append({'dataset':dataset,'id':ident,'row':index,'error':'reference image not found','reference':ref});continue
                    records.append({'dataset':dataset,'id':ident,'smiles':smiles,'reference':ref,'reference_sha256':digest(ref) if ref else None,'input_row':index,'input_sha256':files[-1]['sha256'],'font_families':spec.get('font_families')})
    if not records:raise ValueError('No usable CSV records')
    return records,issues,files
