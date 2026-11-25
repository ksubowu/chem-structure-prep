"""Seeded color styles and literal H -> labeled pseudoatom augmentation."""
import ast,hashlib,random,re,warnings
from pathlib import Path
import numpy as np

SUBSTITUTION_SOURCE=Path(__file__).parent/'assets/labels.json'
PALETTES=[('navy',(0.12,0.24,0.48)),('teal',(0.02,0.36,0.38)),('burgundy',(0.49,0.10,0.20)),('violet',(0.34,0.16,0.52)),('forest',(0.10,0.36,0.16)),('brown',(0.43,0.25,0.08))]
DUMMY_RGB=(0.62,0.12,0.55)


def substitution_inventory(path=None):
    import json
    source=Path(path) if path else SUBSTITUTION_SOURCE
    data=json.loads(source.read_text())
    labels=data['labels'] if isinstance(data,dict) else data
    return {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'labels':[item if isinstance(item,dict) else {'label':item} for item in labels]}


def sample_seed(seed,dataset,ident):return int.from_bytes(hashlib.sha256(f'{seed}|{dataset}|{ident}|color-v1'.encode()).digest()[:8],'big')


def hydrogen_audit(mol):
    target=mol
    if any(b.bondOrder()==4 for b in mol.iterateBonds()):
        target=mol.clone()
        try:target.dearomatize()
        except Exception:pass
    total=0;errors=[]
    for a in target.iterateAtoms():
        if a.symbol()=='H':total+=1
        elif not a.isPseudoatom() and not a.isRSite():
            try:total+=a.countImplicitHydrogens()
            except Exception as exc:errors.append({'atom_id':a.index(),'error':str(exc)})
    return total,errors


def hydrogen_count(mol):return hydrogen_audit(mol)[0]

def heavy_count(mol):return sum(a.symbol()!='H' for a in mol.iterateAtoms())


def prepare(ig,mol,options):
    rng=random.Random(options['seed']);draw=rng.random();requested=draw<options.get('probability',.5)
    before_h,h_errors=hydrogen_audit(mol);before_heavy=heavy_count(mol)
    info={'recipe':'literal_H_to_labeled_dummy_v2','seed':options['seed'],'probability':options.get('probability',.5),'bernoulli_draw':draw,'requested':requested,'applied':False,'source_hydrogen_count':before_h,'source_heavy_atom_count':before_heavy,'label_source':'bundled labels.json snapshot'}
    info['source_hydrogen_count_errors']=h_errors
    if requested and h_errors:
        info['reason']='source_valence_prevents_reliable_H_substitution'
    elif requested:
        candidate=mol.clone();was_aromatic=any(b.bondOrder()==4 for b in candidate.iterateBonds())
        if was_aromatic:candidate.dearomatize()
        candidate.unfoldHydrogens();sites=[]
        for atom in candidate.iterateAtoms():
            if atom.symbol()!='H' or atom.isotope()!=0 or atom.charge() not in (None,0):continue
            neighbors=list(atom.iterateNeighbors())
            if len(neighbors)==1 and neighbors[0].symbol()!='H' and not neighbors[0].isPseudoatom() and neighbors[0].bond().bondOrder()==1:sites.append(atom)
        if sites:
            atom=rng.choice(sites);parent=next(atom.iterateNeighbors());label=rng.choice(options['labels'])
            old_parent_symbol=parent.symbol();sentinel='ChemPrepNewDummy'
            while any(a.symbol()==sentinel for a in candidate.iterateAtoms()):sentinel+='X'
            atom.resetAtom(sentinel);candidate.foldHydrogens()
            # Folding H leaves sparse Indigo atom IDs. Its own CXSMILES reload
            # compacts IDs, preserves stereo and labels, and avoids invalid graph indices.
            candidate=ig.loadMolecule(candidate.smiles())
            # resetAtom('Ac'/'Pr'/...) parses a real element. Encode the label
            # as a CXSMILES pseudoatom alias instead. A molfile roundtrip here
            # would lose some unlaid stereocenters; SMILES retains their @ tags.
            encoded=candidate.smiles();match=re.search(r'\$([^$]*)\$',encoded)
            if match is None or any(c in label for c in '\n\r;$|'):
                raise ValueError('Cannot encode the dummy CXSMILES alias')
            labels=match.group(1).split(';')
            indices=[i for i,value in enumerate(labels) if value==sentinel]
            if len(indices)!=1:raise ValueError('New dummy alias is not unique')
            dummy_id=indices[0];labels[dummy_id]=label
            encoded=encoded[:match.start(1)]+';'.join(labels)+encoded[match.end(1):]
            candidate=ig.loadMolecule(encoded)
            dummy=candidate.getAtom(dummy_id)
            if was_aromatic:candidate.aromatize()
            parent=next(dummy.iterateNeighbors())
            if not (dummy.isPseudoatom() and dummy.degree()==1):raise ValueError('Selected alias is not an opaque, singly bonded pseudoatom')
            if hydrogen_count(candidate)!=before_h-1:raise ValueError('H substitution must remove exactly one H')
            if heavy_count(candidate)!=before_heavy+1:raise ValueError('H substitution must add exactly one pseudoatom')
            info.update({'applied':True,'label':label,'dummy_atom_id':dummy.index(),'parent_atom_id':parent.index(),'parent_symbol':old_parent_symbol,'hydrogens_replaced':1,'reason':'one_unlabeled_H_replaced_by_one_labeled_pseudoatom'})
            mol=candidate
        else:info['reason']='no_eligible_nonisotopic_bonded_H'
    else:info['reason']='bernoulli_not_selected'
    info['rendered_hydrogen_count']=hydrogen_count(mol);info['rendered_heavy_atom_count']=heavy_count(mol)
    palette,rgb=rng.choice(PALETTES);element_colors=rng.random()<.6
    ig.setOption('render-coloring',element_colors);ig.setOption('render-base-color',','.join(map(str,rgb)))
    if info['applied']:
        ig.setOption('render-highlight-color-enabled',True);ig.setOption('render-highlight-color',','.join(map(str,DUMMY_RGB)))
        ig.setOption('render-highlight-thickness-enabled',False);mol.getAtom(info['dummy_atom_id']).highlight()
    color={'palette':palette,'mode':'element_colors_and_tinted_bonds' if element_colors else 'uniform_colored_structure','base_rgb':rgb,'dummy_rgb':DUMMY_RGB if info['applied'] else None}
    return mol,info,color


def parse_rgb(value):
    m=re.fullmatch(r'rgb\(([^)]+)\)',value or '')
    if not m:return None
    return np.array([float(s.strip().rstrip('%'))/(100 if '%' in s else 255) for s in m[1].split(',')])


def assign_dummy_glyphs(glyphs,grouped,augmentation):
    if not augmentation['applied']:return
    owner=augmentation['dummy_atom_id'];expected=np.array(DUMMY_RGB)
    chosen=[g for g in glyphs if (rgb:=parse_rgb(g.get('fill'))) is not None and np.allclose(rgb,expected,atol=1e-5)]
    if not chosen:raise ValueError('Missing highlighted dummy label glyphs')
    boxes={tuple(g['bbox']) for g in chosen}
    for atom_id in grouped:grouped[atom_id]=[b for b in grouped[atom_id] if tuple(b) not in boxes]
    grouped[owner]=[g['bbox'] for g in chosen]
    augmentation['label_glyph_count']=len(chosen);augmentation['label_box_source']='unique_highlight_color_owned_SVG_glyphs'
