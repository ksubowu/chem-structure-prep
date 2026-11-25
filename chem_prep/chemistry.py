"""Molecule loading with explicit support for opaque bracket labels."""
import re
from .backend import Indigo

def load_molecule(ig, smiles):
    if not smiles.strip():raise ValueError('empty SMILES in source CSV')
    try:return ig.loadMolecule(smiles),[]
    except Exception:pass
    # This dataset includes bracketed abbreviations and formula fragments that
    # are labels, rather than standard SMILES atoms. Preserve them as pseudoatoms.
    labels=[];pieces=[];i=0;atom_index=0
    while i<len(smiles):
        c=smiles[i]
        if c=='[':
            j=i+1;depth=1
            while j<len(smiles) and depth:
                if smiles[j]=='[':depth+=1
                elif smiles[j]==']':depth-=1
                j+=1
            if depth:raise ValueError('unbalanced bracket in source SMILES')
            token=smiles[i:j];label=token[1:-1]
            custom=bool(re.fullmatch(r"R(?:\d+|['’]+|[a-zA-Z]+)?",label)) or any(ord(x)>127 for x in label)
            if not custom:
                try:
                    parsed=ig.loadMolecule(token)
                    if parsed.countAtoms()!=1:custom=True
                except Exception:custom=True
            if custom:
                display=label.replace('’',"'")
                labels.append((atom_index,display,label));pieces.append('[C]')
            else:pieces.append(token)
            atom_index+=1;i=j
        elif c.isalpha() or c=='*':
            token=smiles[i:i+2] if smiles[i:i+2] in ('Cl','Br') else c
            pieces.append(token);i+=len(token);atom_index+=1
        else:pieces.append(c);i+=1
    mol=ig.loadMolecule(''.join(pieces))
    for index,display,original in labels:mol.getAtom(index).resetAtom(display)
    return mol,[{'atom_id':index,'source_label':original,'display_label':display} for index,display,original in labels]

