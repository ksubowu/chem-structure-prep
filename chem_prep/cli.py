"""Public CLI. CPU thread limits are set before importing numerical libraries."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,ast,json,shutil,sys,tarfile,warnings
from pathlib import Path


def extract_labels(source,output):
    import hashlib,re
    text=Path(source).read_text()
    with warnings.catch_warnings():warnings.simplefilter('ignore',SyntaxWarning);tree=ast.parse(text)
    labels={}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Substitution' or len(node.args)<3:continue
        try:
            names=ast.literal_eval(node.args[0]);smarts=ast.literal_eval(node.args[1]);smiles=ast.literal_eval(node.args[2])
        except (ValueError,TypeError,SyntaxError):continue
        for name in names:
            if re.fullmatch(r'[A-Za-z][A-Za-z0-9]{1,19}',name) and any(c.isdigit() for c in name):labels.setdefault(name,{'label':name,'example_smarts':smarts,'example_smiles':smiles})
    if not labels:raise ValueError('No literal alphanumeric Substitution aliases found')
    from .io import write_json
    write_json(output,{'source':Path(source).name,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'labels':[labels[k] for k in sorted(labels)]})
    return {'labels':len(labels),'output':str(output)}


def main(argv=None):
    parser=argparse.ArgumentParser(prog='chem-prepare',description='CSV/SMILES -> chemical images, annotations and balanced train/val datasets')
    parser.add_argument('--version',action='version',version='chem-prepare '+__import__('chem_prep').__version__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Check bundled backend, pixel coords and fonts')
    init=sub.add_parser('init',help='Copy a self-contained example job');init.add_argument('directory',type=Path)
    for cmd in ['run','render','split']:
        p=sub.add_parser(cmd);p.add_argument('--config',required=True,type=Path)
    for cmd in ['validate','review']:
        p=sub.add_parser(cmd);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('balance',help='Train-only augmentation/balancing with frozen validation');p.add_argument('--dataset',required=True,type=Path);p.add_argument('--charged-fraction',type=float,default=.25);p.add_argument('--charge-variants',type=int);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261114)
    p=sub.add_parser('extract-labels',help='Read Substitution aliases via AST without importing the source script');p.add_argument('source',type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('pack',help='Archive a validated portable dataset');p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('audit-bands',help='Audit stored SVG bands and export review candidates without changing existing datasets');p.add_argument('--manifest',required=True,type=Path,nargs='+');p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('probe',help='Export exclusive atom/group/bond dual-color probes and pixel masks');p.add_argument('--smiles',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--atom-ids');p.add_argument('--bond-ids');p.add_argument('--group-smarts');p.add_argument('--match-index',type=int,default=0);p.add_argument('--all-atoms',action='store_true');p.add_argument('--all-bonds',action='store_true');p.add_argument('--colored',action='store_true');p.add_argument('--font',default='DejaVu Sans');p.add_argument('--size',type=int,default=640);p.add_argument('--include-internal-bonds',action='store_true')
    p=sub.add_parser('add-lines',help='Copy an unsplit dataset and add per-bond center-to-center line targets');p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--endpoint-mode',choices=['atom_center','box_center'],default='atom_center')
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            from .backend import doctor
            result=doctor()
        elif args.command=='add-lines':
            from .lines import add_dataset
            result=add_dataset(args.dataset,args.output,args.endpoint_mode)
        elif args.command=='probe':
            from .probe import export
            indices=lambda v:None if v is None else [int(i) for i in v.split(',')]
            result=export(args.smiles,args.output,indices(args.atom_ids),indices(args.bond_ids),args.group_smarts,args.match_index,args.all_atoms,args.all_bonds,args.colored,args.font,args.size,args.include_internal_bonds)
        elif args.command=='audit-bands':
            from .band_audit import audit
            result=audit(args.manifest,args.output,args.workers)
        elif args.command=='init':
            # Example resources live inside the package and survive pip installation.
            source=Path(__file__).parent/'assets/example';args.directory.mkdir(parents=True,exist_ok=True)
            for name in ['demo.csv','recipe.yaml']:
                if (args.directory/name).exists():raise FileExistsError(args.directory/name)
            for name in ['demo.csv','recipe.yaml']:shutil.copy2(source/name,args.directory/name)
            result={'config':str(args.directory/'recipe.yaml'),'next':f'chem-prepare run --config {args.directory/"recipe.yaml"}'}
        elif args.command=='extract-labels':result=extract_labels(args.source,args.output)
        elif args.command in ['run','render','split']:
            from .config import load
            from . import pipeline
            cfg=load(args.config)
            if args.command=='run':result=pipeline.run(cfg)
            elif args.command=='render':result={'rendered':len(pipeline.render(cfg)),'output':cfg['output']}
            else:
                train,val=pipeline.split(cfg);result={'train':len(train),'val':len(val),'output':cfg['output']}
        elif args.command=='validate':
            from .validate import validate
            result,_=validate(args.dataset,args.workers)
        elif args.command=='review':
            from .review import build
            result=build(args.dataset)
        elif args.command=='balance':
            from .io import read_json
            from .pipeline import finish
            root=args.dataset.resolve();cfg=read_json(root/'run_metadata.json')['resolved_configuration'];cfg['output']=str(root);cfg['workers']=args.workers
            cfg['balance'].update({'enabled':True,'charged_fraction':args.charged_fraction,'seed':args.seed})
            if args.charge_variants is not None:cfg['charge_augmentation']['count']=args.charge_variants
            result=finish(cfg)
        elif args.command=='pack':
            from .validate import validate
            validate(args.dataset,args.workers);root=args.dataset.resolve();dest=args.output.resolve()
            if dest.exists():raise FileExistsError(dest)
            if dest.is_relative_to(root):raise ValueError('Archive output must be outside dataset directory')
            with tarfile.open(dest,'w:gz') as archive:archive.add(root,arcname=root.name)
            result={'archive':str(dest),'bytes':dest.stat().st_size}
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
