#!/usr/bin/env python3
"""Render PubChem SMILES and extract renderer-owned bond instances and label boxes.

Masks come from the same SVG primitives as the source PNG, never from dilated
centerlines. NPZ ink/region arrays have shape [bond_count, height, width].
This prepares data and validates annotations; it does not train a model.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D

ATOM_CLASSES = ['other', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'H', 'Si']
CHARGE_CLASSES = {-4: 13, -2: 14, -1: 15, 1: 16, 2: 17}
BOND_CLASSES = ['other', 'single', 'double', 'triple', 'aromatic', 'wedge', 'dash']
SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    return rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, addChiralHs=True)


def bond_class(bond):
    direction = bond.GetBondDir()
    if direction == Chem.BondDir.BEGINWEDGE:
        return 5
    if direction == Chem.BondDir.BEGINDASH:
        return 6
    return {1.: 1, 2.: 2, 3.: 3, 1.5: 4}.get(bond.GetBondTypeAsDouble(), 0)


def svg_bytes(root):
    return ET.tostring(root, encoding='utf-8')


def raster(root):
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_bytes(root)))).convert('RGBA')


def mask_for(root, elements):
    isolated = ET.Element(root.tag, root.attrib)
    for el in elements:
        isolated.append(copy.deepcopy(el))
    return np.asarray(raster(isolated))[:, :, 3] > 0


def bbox(mask, padding=1):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty renderer mask')
    h, w = mask.shape
    x0, y0 = max(0, int(x.min())-padding), max(0, int(y.min())-padding)
    x1, y1 = min(w, int(x.max())+1+padding), min(h, int(y.max())+1+padding)
    return [x0, y0, x1-x0, y1-y0]


def charge_nodes(nodes, q):
    chosen = [el for el in nodes if re.search(r'[+\-−]', ''.join(el.itertext()))]
    if abs(q) > 1 and chosen:
        # RDKit can emit the magnitude and sign as separate superscript glyphs.
        anchor = chosen[0]
        for el in nodes:
            text = ''.join(el.itertext()).strip()
            if text == str(abs(q)) and abs(float(el.get('y'))-float(anchor.get('y'))) < 1:
                chosen.append(el)
    return chosen


def render_record(row, csv_path, output, size):
    mol = prepare(row['SMILES'])
    if any(a.GetFormalCharge() not in {0, *CHARGE_CLASSES} for a in mol.GetAtoms()):
        raise ValueError('Charge outside existing detector class schema')
    cid = str(row['pubchem_cid'])
    stem = f"{csv_path.stem}_row{row['_row']}_cid{cid}"
    # Text SVG exposes the exact atom/charge glyph ownership. All PNGs and masks
    # are rasterized from this one SVG with the same Cairo/font implementation.
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size, -1, -1, True)
    opts = drawer.drawOptions()
    # Freeze this prepared graph. A second implicit preparation can insert
    # stereo hydrogens that would otherwise lack atom boxes in our annotation.
    opts.prepareMolsBeforeDrawing = False
    opts.useBWAtomPalette()
    opts.fixedBondLength = 32
    opts.fixedFontSize = 18
    opts.bondLineWidth = 2
    opts.padding = .06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    root = ET.fromstring(svg)
    original = raster(root).convert('RGB')
    original.save(output/'images'/f'{stem}.png')
    (output/'svg'/f'{stem}.svg').write_text(svg, encoding='utf-8')
    centers = [list(map(float, (drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y)))
               for i in range(mol.GetNumAtoms())]
    atoms, charges, bonds, inks, regions = [], [], [], [], []
    glyph_masks = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        nodes = [el for el in root if el.tag.endswith('text') and el.get('class') == f'atom-{i}']
        q = atom.GetFormalCharge()
        qnodes = charge_nodes(nodes, q) if q else []
        atom_nodes = [el for el in nodes if el not in qnodes]
        if nodes:
            glyph_masks.append(mask_for(root, nodes))
        if atom_nodes:
            box = bbox(mask_for(root, atom_nodes), padding=2)
            box_source = 'rendered_atom_label_ink'
        else:
            x, y = centers[i]
            half = 6
            x0, y0 = max(0, int(x)-half), max(0, int(y)-half)
            x1, y1 = min(size, int(x)+half+1), min(size, int(y)+half+1)
            box = [x0, y0, x1-x0, y1-y0]
            box_source = 'implicit_atom_anchor'
        symbol = atom.GetSymbol()
        atoms.append(dict(index=i, symbol=symbol,
                          category_id=ATOM_CLASSES.index(symbol) if symbol in ATOM_CLASSES else 0,
                          center=centers[i], bbox=box, bbox_source=box_source,
                          formal_charge=q, isotope=atom.GetIsotope()))
        if q:
            if not qnodes:
                raise ValueError(f'Missing charge glyph for atom {i}')
            charges.append(dict(atom_index=i, category_id=CHARGE_CLASSES[q],
                                label=f'{q:+d}', canonical_label=f'{q:+d}',
                                bbox=bbox(mask_for(root, qnodes), padding=2)))
    for bond in mol.GetBonds():
        i = bond.GetIdx()
        elements = [el for el in root if f'bond-{i}' in el.get('class', '').split()]
        if not elements:
            raise ValueError(f'No SVG primitives for bond {i}')
        ink = mask_for(root, elements)
        y, x = np.nonzero(ink)
        if not len(x):
            raise ValueError(f'Empty bond {i}')
        hull = cv2.convexHull(np.column_stack((x, y)).astype(np.int32))
        region = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(region, hull, 1)
        region = region.astype(bool) | ink
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        category = bond_class(bond)
        bonds.append(dict(index=i, begin=begin, end=end, category_id=category,
                          type=BOND_CLASSES[category], chemical_bond_type=str(bond.GetBondType()),
                          is_aromatic=bond.GetIsAromatic(), stereo=str(bond.GetStereo()),
                          bond_dir=str(bond.GetBondDir()), line=centers[begin]+centers[end],
                          direction=(dict(tip_atom=begin, wide_atom=end) if category in (5, 6) else None),
                          ink_bbox=bbox(ink, 0), ink_area=int(ink.sum()),
                          region_polygon=hull.reshape(-1, 2).tolist(), region_area=int(region.sum()),
                          mask_index=i, svg_primitive_count=len(elements)))
        inks.append(ink)
        regions.append(region)
    inks = np.stack(inks) if inks else np.zeros((0, size, size), bool)
    regions = np.stack(regions) if regions else np.zeros_like(inks)
    np.savez_compressed(output/'masks'/f'{stem}.npz', ink=inks, region=regions)
    annotation = dict(schema='bond_band_v1', image_id=stem, image_size=[size, size],
                      source=dict(csv=str(csv_path), row=row['_row'], pubchem_cid=cid,
                                  smiles=row['SMILES']),
                      canonical_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(row['SMILES'])),
                      rendered_molfile=Chem.MolToMolBlock(mol),
                      image=f'images/{stem}.png', svg=f'svg/{stem}.svg', masks=f'masks/{stem}.npz',
                      bbox_format='xywh_pixels', mask_definition=dict(
                          ink='All nonzero-alpha pixels of the actual SVG bond primitives.',
                          region='Convex envelope of the bond ink; includes stroke gaps. Not an ink mask.',
                          overlaps='Different instances may overlap; masks are not an exclusive label map.'),
                      atoms=atoms, charges=charges, bonds=bonds)
    # Smoke gates compare raster ownership, instance structure, graph and file IO.
    reloaded = Chem.MolFromMolBlock(annotation['rendered_molfile'], removeHs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(reloaded)) == annotation['canonical_smiles'], 'Graph/stereo roundtrip mismatch'
    assert len(atoms) == mol.GetNumAtoms() and len(bonds) == mol.GetNumBonds()
    assert len(charges) == sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    for obj in atoms+charges:
        x, y, w, h = obj['bbox']
        assert w > 0 and h > 0 and 0 <= x < x+w <= size and 0 <= y < y+h <= size
    assert np.all(~inks | regions)
    ink_union = inks.any(0)
    original_ink = np.asarray(original).min(2) < 245
    # Label masks account for atom symbols, hydrogens, isotopes and charges.
    assigned = ink_union | (np.stack(glyph_masks).any(0) if glyph_masks else False)
    coverage = float((assigned & original_ink).sum()/max(1, original_ink.sum()))
    precision = float((ink_union & original_ink).sum()/max(1, ink_union.sum()))
    # Nonzero-alpha masks deliberately include antialiasing fringes. Most pixels
    # must still be visible dark ink. Tiny classless RDKit join paths are audited.
    assert coverage > .98, f'Unassigned rendered ink: {coverage:.4f}'
    assert precision > .75, f'Bond mask misalignment: {precision:.4f}'
    with np.load(output/annotation['masks']) as stored:
        assert np.array_equal(stored['ink'], inks)
    annotation['audit'] = dict(status='passed', assigned_dark_ink_fraction=coverage,
                               bond_mask_dark_pixel_fraction=precision,
                               graph_and_stereo_roundtrip=True,
                               implicit_atom_boxes=sum(a['bbox_source']=='implicit_atom_anchor' for a in atoms))
    write_json(output/'annotations'/f'{stem}.json', annotation)
    return annotation, original, inks, regions


def comparison(annotation, original, inks, regions, path):
    size = original.width
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 12)
    atom_img = original.copy()
    draw = ImageDraw.Draw(atom_img)
    for a in annotation['atoms']:
        x, y, w, h = a['bbox']
        draw.rectangle((x, y, x+w, y+h), outline='#147cce', width=2)
    for q in annotation['charges']:
        x, y, w, h = q['bbox']
        draw.rectangle((x, y, x+w, y+h), outline='#d82060', width=2)
    band_img = np.asarray(original).astype(float).copy()
    colors = []
    for i in range(len(inks)):
        color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(i*137.5)%180, 220, 205]]]), cv2.COLOR_HSV2RGB)[0, 0])
        colors.append(color)
        band_img[regions[i]] = .84*band_img[regions[i]] + .16*np.array(color)
        band_img[inks[i]] = .32*band_img[inks[i]] + .68*np.array(color)
    band_img = Image.fromarray(band_img.clip(0, 255).astype('uint8'))
    bdraw = ImageDraw.Draw(band_img)
    for b, color in zip(annotation['bonds'], colors):
        poly = [tuple(p) for p in b['region_polygon']]
        bdraw.line(poly+[poly[0]], fill=color, width=1)
        if b['direction']:
            x, y = annotation['atoms'][b['begin']]['center']
            bdraw.ellipse((x-3,y-3,x+3,y+3), fill=color)
    margin, header, footer = 18, 58, 74
    canvas = Image.new('RGB', (3*size+4*margin, size+header+footer), '#f1f4f8')
    draw = ImageDraw.Draw(canvas)
    titles = ['Original', 'Atom + charge boxes', 'Bond bands + original']
    for col, (im, title) in enumerate(zip([original, atom_img, band_img], titles)):
        x = margin+(size+margin)*col
        draw.text((x, 18), title, font=font, fill='#182535')
        canvas.paste(im, (x, header))
    draw.text((margin, size+header+12),
              f"PubChem CID {annotation['source']['pubchem_cid']}  |  "
              f"{len(annotation['atoms'])} atoms, {len(annotation['charges'])} charges, {len(inks)} bonds",
              font=font, fill='#182535')
    draw.text((margin, size+header+42),
              'Blue: atom boxes; pink: charge boxes. Bands: colored actual ink + pale convex envelope; dot: wedge/dash tip.',
              font=small, fill='#33445a')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--scan', type=int, default=1000)
    parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args()
    assert args.limit > 0 and args.scan >= args.limit and args.size >= 256
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ['images','svg','annotations','masks','comparisons']:
        (args.output/directory).mkdir()
    candidates, rejected = [], []
    with args.csv.open(newline='') as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= args.scan:
                break
            row['_row'] = index
            try:
                mol = prepare(row['SMILES'])
                n = mol.GetNumAtoms()
                if not 5 <= n <= 55:
                    continue
                if any(a.GetFormalCharge() not in {0, *CHARGE_CLASSES} for a in mol.GetAtoms()):
                    continue
                types = {bond_class(b) for b in mol.GetBonds()}
                q = sum(a.GetFormalCharge()!=0 for a in mol.GetAtoms())
                # Feature-rich moderate-sized molecule for the requested comparison.
                score = 12*(5 in types)+12*(6 in types)+20*(q>0)+5*(q>1)+3*(2 in types)+3*(3 in types)-abs(n-24)/20
                candidates.append((score,row,types,q))
            except Exception as exc:
                rejected.append(dict(row=index,error=str(exc)))
    candidates.sort(key=lambda c:(-c[0],c[1]['_row']))
    selected = candidates[:1]
    # Add explicit style coverage before ordinary sequential CSV rows.
    for category in [1,2,3,4,5,6]:
        match = next((c for c in candidates if category in c[2] and c not in selected), None)
        if match is not None and len(selected)<args.limit:
            selected.append(match)
    for c in sorted(candidates,key=lambda c:c[1]['_row']):
        if len(selected)>=args.limit:
            break
        if c not in selected:
            selected.append(c)
    assert len(selected)==args.limit, 'Not enough eligible molecules'
    manifest, counts, audits = [], Counter(), []
    for _,row,_,_ in selected:
        annotation, image, inks, regions = render_record(row,args.csv.resolve(),args.output,args.size)
        stem = annotation['image_id']
        comparison(annotation,image,inks,regions,args.output/'comparisons'/f'{stem}.png')
        for b in annotation['bonds']:
            counts[b['type']]+=1
        audits.append(annotation['audit'])
        canonical = annotation['canonical_smiles']
        manifest.append(dict(image=annotation['image'],truth=f'annotations/{stem}.json',
                             masks=annotation['masks'],pubchem_cid=row['pubchem_cid'],
                             source_row=row['_row'],group=hashlib.sha256(canonical.encode()).hexdigest(),
                             split='smoke_only',source='pubchem_renderer_svg'))
        print(f"PASS CID {row['pubchem_cid']}: {len(annotation['atoms'])} atoms, "
              f"{len(annotation['charges'])} charges, {len(annotation['bonds'])} bonds",flush=True)
    sample=manifest[0]
    Image.open(args.output/'comparisons'/Path(sample['image']).name).save(args.output/'sample_comparison.png')
    write_json(args.output/'manifest.json',manifest)
    report=dict(status='passed',molecules=len(manifest),bond_counts=dict(counts),
                total_atoms=sum(len(json.loads((args.output/r['truth']).read_text())['atoms']) for r in manifest),
                total_charges=sum(len(json.loads((args.output/r['truth']).read_text())['charges']) for r in manifest),
                min_assigned_dark_ink_fraction=min(a['assigned_dark_ink_fraction'] for a in audits),
                min_bond_mask_dark_pixel_fraction=min(a['bond_mask_dark_pixel_fraction'] for a in audits),
                rdkit_version=rdBase.rdkitVersion,cairo_version=cairosvg.__version__,
                source_csv=str(args.csv.resolve()),scan_limit=args.scan,
                sample=sample,scan_rejections=rejected,
                notes=['Smoke dataset only; no training or held-out evaluation split assigned.',
                       'Stereo hydrogens inserted by RDKit preparation are explicit annotated H atoms.',
                       'Aromatic chemical flags retained; bond class describes the drawn Kekule stroke.',
                       'Implicit carbon boxes are geometric anchors, not visible glyph boxes.',
                       'Envelope is a convex hull; actual disconnected stroke masks are stored separately.'])
    write_json(args.output/'smoke_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
