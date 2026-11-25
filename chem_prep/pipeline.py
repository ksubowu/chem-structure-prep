"""Configurable render -> grouped split -> train-only augmentation -> balance workflow."""
import copy,hashlib,json,platform,random
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import quote
import numpy as np
from . import __version__
from .config import ingest
from .io import asset_stem,digest,read_json,seed_for,write_json,write_manifest
from .augmentation import substitution_inventory
from .render import render_one
from .split import assign
from .balance import apply,snapshot_validation,verify_holdout
from .validate import audit_assets,validate
from .charges import generate,FONTS
from .review import build as build_review
from .lines import extend_paths


def parallel(function,jobs,workers):
    import time
    total=len(jobs);started=time.monotonic();results=[]
    if workers==1:iterator=map(function,jobs);pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
        iterator=pool.map(function,jobs,chunksize=4)
    try:
        for i,result in enumerate(iterator,1):
            results.append(result)
            if total>=100 and (i%100==0 or i==total):
                print(f'{function.__name__}: {i}/{total}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        if pool is not None:pool.shutdown()
    return results


def balanced_selection(records,count,seed):
    if count>=len(records):return list(records)
    rng=random.Random(seed);pools=defaultdict(list)
    for r in records:pools[r['dataset']].append(r)
    for p in pools.values():rng.shuffle(p)
    selected=[]
    while len(selected)<count:
        for name in sorted(pools):
            if pools[name] and len(selected)<count:selected.append(pools[name].pop())
    return sorted(selected,key=lambda r:(r['dataset'],r['id']))


def semantic_config(cfg,records,inventory):
    c={k:copy.deepcopy(cfg[k]) for k in ['render','color','split','charge_augmentation','balance','quality','seed']}
    c['color']['labels_file']=inventory['source_sha256']
    c['charge_augmentation']['fonts']=[{'name':Path(p).name,'sha256':digest(p)} for p in (cfg['charge_augmentation']['fonts'] or FONTS)]
    c['inputs']=[{k:r[k] for k in ['dataset','id','smiles','reference_sha256','input_row','input_sha256','font_families']} for r in records]
    c['tool_version']=__version__
    c['band_assignment_revision']=3
    c['atom_assignment_revision']=2
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def render(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    records,issues,files=ingest(cfg);inventory=substitution_inventory(cfg['color']['labels_file']);labels=[r['label'] for r in inventory['labels']]
    if not labels and cfg['color']['dummy_probability']>0 and cfg['color']['count']:raise ValueError('No labels for dummy augmentation')
    signature=semantic_config(cfg,records,inventory);state_path=root/'run_metadata.json'
    if state_path.exists() and read_json(state_path)['config_fingerprint']!=signature:raise ValueError('Output already belongs to a different recipe/input snapshot; select another output directory')
    if not state_path.exists():
        from .backend import doctor
        write_json(state_path,{'version':__version__,'config_fingerprint':signature,'inputs':files,'resolved_configuration':cfg,'environment':doctor()})
    write_json(root/'label_inventory.json',inventory);write_json(root/'input_issues.json',issues)
    jobs=[]
    selected=balanced_selection(records,min(cfg['color']['count'],len(records)),cfg['seed'])
    for variant,collection,is_color in [('bw',records,False),('color',selected,True)]:
        for source in collection:
            stem=asset_stem(source['dataset'],source['id'],source['smiles']);settings=copy.deepcopy(cfg['render'])
            if source.get('font_families'):settings['font_families']=source['font_families']
            paths={key:f'assets/{variant}/{folder}/{stem}.{ext}' for key,folder,ext in [('image','images','png'),('annotation','annotations','json'),('masks','masks','npz'),('band_target','targets','json'),('svg','svg','svg'),('overlay','overlays','png'),('view_check','view_check','jpg')]}
            paths.update(extend_paths(paths))
            sample_id=f'{variant}/{quote(source["dataset"],safe="")}/{quote(source["id"],safe="")}';parent_id=f'bw/{quote(source["dataset"],safe="")}/{quote(source["id"],safe="")}' if is_color else None
            seed=seed_for(cfg['seed'],sample_id)
            fingerprint=hashlib.sha256(json.dumps({'recipe':signature,'sample_id':sample_id,'seed':seed},sort_keys=True).encode()).hexdigest()
            jobs.append({'source':source,'variant':variant,'sample_id':sample_id,'parent_id':parent_id,'is_color':is_color,'seed':seed,'root':str(root),'paths':paths,'render':settings,'labels':labels,'dummy_probability':cfg['color']['dummy_probability'],'fingerprint':fingerprint})
    old={r['sample_id']:r for r in read_json(root/'render_manifest.json')} if (root/'render_manifest.json').exists() else {}
    checkpoint_paths=list((root/'_render_checkpoints').glob('*.json'))
    if checkpoint_paths:
        with ThreadPoolExecutor(max_workers=12) as checkpoint_pool:
            for prior in checkpoint_pool.map(read_json,checkpoint_paths):
                if prior.get('status')=='ok':old[prior['sample_id']]=prior
    reused=[];pending=[]
    for job in jobs:
        prior=old.get(job['sample_id'])
        if prior and prior.get('status')=='ok' and prior.get('job_fingerprint')==job['fingerprint'] and all((root/prior[p]).is_file() for p in job['paths']):reused.append(prior)
        else:pending.append(job)
    print(f'Rendering {len(pending)} images; reusing {len(reused)}',flush=True)
    results=reused+parallel(render_one,pending,cfg['workers']);results.sort(key=lambda r:r['sample_id']);write_manifest(root,'render_manifest',results)
    success=[r for r in results if r['status']=='ok'];errors=[r for r in results if r['status']!='ok'];write_json(root/'render_errors.json',errors)
    # If a BW draw fails but its color succeeds, parent linkage still uses
    # source chemistry. Parent_id is retained as provenance, not a dangling GT index.
    if not success:raise RuntimeError('No images rendered successfully')
    if (issues or errors) and cfg['quality']['fail_on_render_error']:raise RuntimeError('Input/render errors quarantined; fail_on_render_error is enabled')
    write_json(root/'render_summary.json',{'ok':len(success),'input_issues':len(issues),'render_errors':len(errors),'color':sum(r['is_color'] for r in success),'dummy_applied':sum(r['dummy_applied'] for r in success),'charged':sum(r['has_charge'] for r in success)})
    if not (root/'data_manifest.json').exists() or not (root/'source_train.json').exists():write_manifest(root,'data_manifest',success)
    return success


def split(cfg,records=None):
    root=Path(cfg['output']);records=records or [r for r in read_json(root/'render_manifest.json') if r['status']=='ok']
    if (root/'val.json').exists():
        base=read_json(root/'source_train.json')+read_json(root/'val.json')
        if {r['sample_id'] for r in base}!={r['sample_id'] for r in records}:raise RuntimeError('Inputs changed after the holdout was frozen')
        verify_holdout(root);return read_json(root/'source_train.json'),read_json(root/'val.json')
    report,_=audit_assets(root,records,cfg['workers']);write_json(root/'pre_split_validation.json',report)
    if report['failures']:raise RuntimeError('Cannot split invalid render assets; inspect pre_split_validation.json')
    train,val,report=assign(root,copy.deepcopy(records),cfg['split']);write_json(root/'source_train.json',train);write_json(root/'unique_train.json',train);write_manifest(root,'data_manifest',train+val)
    snapshot_validation(root)
    return train,val


def augment_charges(root,settings,workers):
    root=Path(root);original=read_json(root/'source_train.json');existing_path=root/'charge_manifest.json'
    fingerprint=hashlib.sha256(json.dumps({**settings,'fonts':[{'name':Path(p).name,'sha256':digest(p)} for p in (settings['fonts'] or FONTS)]},sort_keys=True).encode()).hexdigest()
    if existing_path.exists():
        cached=read_json(root/'charge_stage.json')
        if cached['fingerprint']!=fingerprint and cached.get('actual',0)>0:raise RuntimeError('Charge augmentation recipe changed after nonempty generation; select a new dataset output')
        prior=read_json(existing_path)
        if cached['fingerprint']==fingerprint and all(all((root/r[k]).is_file() for k in ['image','annotation','svg','masks','band_target','overlay','view_check']) for r in prior):return prior
    eligible=[r for r in original if r['has_charge'] and all(abs(q)==1 for q in r['charge_values'])]
    requested=settings['count'];count=requested if eligible else 0
    if requested and not eligible:print('No eligible ±1 training source: typography augmentation skipped; multivalent source images remain available for sampling',flush=True)
    jobs=[]
    for n in range(count):
        i=n%len(eligible);rep=n//len(eligible);jobs.append((eligible[i],str(root),str(root),i,rep,seed_for(settings['seed'],eligible[i]['sample_id'],rep),settings))
    results=parallel(generate,jobs,workers);bad=[r for r in results if r['status']!='ok']
    for attempt in range(3):
        if not bad:break
        replacements=[]
        for r in bad:
            i,rep=r['index'],r['rep'];replacements.append((eligible[i],str(root),str(root),i,rep,seed_for(settings['seed'],eligible[i]['sample_id'],rep,'retry',attempt),settings))
        results=[r for r in results if r['status']=='ok']+parallel(generate,replacements,workers);bad=[r for r in results if r['status']!='ok']
    write_json(root/'charge_errors.json',bad)
    if bad:raise RuntimeError('Charge typography generation failed; inspect charge_errors.json')
    results.sort(key=lambda r:r['sample_id']);audit,_=audit_assets(root,results,workers);write_json(root/'charge_augmentation_validation.json',audit)
    if audit['failures']:raise RuntimeError('Charge variants failed independent audit')
    write_json(existing_path,results);write_json(root/'charge_stage.json',{'fingerprint':fingerprint,'requested':requested,'actual':len(results),'eligible_training_parents':len(eligible),'limitations':'Typography replacement currently supports ±1 only; baseline boxes support multivalent charges'})
    return results


def finish(cfg):
    root=Path(cfg['output']);train=read_json(root/'source_train.json');val=read_json(root/'val.json');variants=augment_charges(root,cfg['charge_augmentation'],cfg['workers'])
    unique=train+variants
    if cfg['balance']['enabled']:apply(root,unique,cfg['balance'])
    else:write_manifest(root,'train',unique);write_json(root/'unique_train.json',unique)
    write_manifest(root,'data_manifest',unique+val)
    config={'schema':'chem_structure_prep_dataset_v1','tool_version':__version__,'dataset_root':'.','train_manifest':'train.json','val_manifest':'val.json','unique_assets_manifest':'data_manifest.json','bbox_format':'xyxy_pixels','band_target_schema':'independent_band_quad_v1','line_target_schema':'independent_bond_line_v1','line_endpoint_mode':'atom_center','preserve_rgb':True,'charge_balance':cfg['balance'],'path_root':'all artifact paths are relative to dataset_root','sampling':'uniform shuffle of explicit train entries; do not deduplicate asset_sample_id or add a second weighted sampler'}
    write_json(root/'dataset_config.json',config)
    report,_=validate(root,cfg['workers'])
    if cfg['review']['enabled']:build_review(root)
    return report


def run(cfg):
    records=render(cfg);split(cfg,records);return finish(cfg)
