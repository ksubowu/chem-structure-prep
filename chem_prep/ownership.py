"""Native atom/bond ownership from exclusive, geometry-checked dual-color probes.

Only the selected native objects are highlighted. Color A/B differ exclusively
at the selection; ordinary colored oxygen, aliases, etc. cannot contaminate it.
The production SVG is never replaced by a diagnostic rendering.
"""
import copy,io,json,xml.etree.ElementTree as ET
import cairosvg
import numpy as np
from PIL import Image
from .svg import svg_geometry,matrix,NS
COLORS=((1.,0.,0.),(0.,1.,1.))


def geometry_check(reference,candidate,reference_geometry=None):
    ar,br=ET.fromstring(reference),ET.fromstring(candidate)
    if any(ar.get(k)!=br.get(k) for k in ('viewBox','width','height')):raise ValueError('Highlight changed SVG canvas')
    ag,ap=reference_geometry if reference_geometry is not None else svg_geometry(reference)
    bg,bp=svg_geometry(candidate)
    if len(ag)!=len(bg) or len(ap)!=len(bp):raise ValueError('Highlight changed SVG primitive count')
    for a,b in zip(ag,bg):
        if a['bbox'].shape!=b['bbox'].shape or not np.allclose(a['bbox'],b['bbox'],rtol=0,atol=1e-7):raise ValueError('Highlight moved a glyph')
    for a,b in zip(ap,bp):
        if a['d']!=b['d'] or not np.allclose(a['matrix'],b['matrix'],rtol=0,atol=1e-7):raise ValueError('Highlight moved a bond/outline')
        for k in ('stroke-width','stroke-linecap','stroke-linejoin','fill-rule'):
            if a['style'].get(k)!=b['style'].get(k):raise ValueError('Highlight changed stroke geometry')
    return (ag,ap),(bg,bp)


class OwnershipProbe:
    def __init__(self,ig,renderer,mol,baseline_svg):
        self.ig=ig;self.renderer=renderer;self.mol=mol;self.baseline=baseline_svg
        self.atoms=list(mol.iterateAtoms());self.bonds=list(mol.iterateBonds())
        self.flags=[obj.isHighlighted() for obj in self.atoms+self.bonds]
        self.options={k:ig.getOptionBool(k) for k in ('render-highlight-color-enabled','render-highlight-thickness-enabled','render-highlighted-labels-visible')}
        self.color=json.loads(ig.getOption('render-highlight-color'))
        self.coords=np.array([a.coords() for a in self.atoms])
        self.glyphs,self.paths=svg_geometry(baseline_svg)
        self.render_calls=0;self.hidden_carbon_cache={}

    def __enter__(self):
        for obj in self.atoms+self.bonds:obj.unhighlight()
        self.ig.setOption('render-highlight-color-enabled',True)
        self.ig.setOption('render-highlight-thickness-enabled',False)
        self.ig.setOption('render-highlighted-labels-visible',False)
        return self

    def __exit__(self,*exc):
        for obj,flag in zip(self.atoms+self.bonds,self.flags):
            obj.highlight() if flag else obj.unhighlight()
        for k,value in self.options.items():self.ig.setOption(k,value)
        self.ig.setOption('render-highlight-color',','.join(map(str,self.color)))

    def select(self,atom_ids=(),bond_ids=(),keep_svg=False):
        atom_ids=set(atom_ids);bond_ids=set(bond_ids)
        if not atom_ids<=set(range(len(self.atoms))) or not bond_ids<=set(range(len(self.bonds))):raise ValueError('Selection index out of range')
        # Clear every previous selection, including any preexisting dummy tint.
        for obj in self.atoms+self.bonds:obj.unhighlight()
        hidden=[]
        # Some crowded bridge carbons are already explicitly printed as CH2.
        # Probe visibility instead of inferring it from the chemical symbol.
        for i in sorted(atom_ids):
            a=self.atoms[i]
            plain=(self.ig.getOption('render-label-mode')=='hetero' and a.symbol()=='C' and not a.isPseudoatom() and not a.charge() and not a.isotope() and not a.radical() and a.degree()>0)
            if plain and i not in self.hidden_carbon_cache:
                for obj in self.atoms+self.bonds:obj.unhighlight()
                a.highlight();self.ig.setOption('render-highlight-color','1,0,0')
                trial=bytes(self.renderer.renderToBuffer(self.mol)).decode();self.render_calls+=1
                try:geometry_check(self.baseline,trial,(self.glyphs,self.paths));self.hidden_carbon_cache[i]=False
                except ValueError:
                    old,_=svg_geometry(self.baseline);new,_=svg_geometry(trial)
                    displacement=np.array([a.coords() for a in self.atoms])-self.coords
                    shift=np.median(displacement,axis=0)
                    if not np.allclose(displacement,shift,rtol=0,atol=1e-3):raise
                    x,y=self.coords[i]
                    contains=lambda b:b[0]-1<=x<=b[2]+1 and b[1]-1<=y<=b[3]+1
                    # Revealing C/CHn can also trigger automatic repositioning
                    # of OTHER H labels. The entire trial is discarded. Only
                    # accept a proxy when the trial introduced a NEW glyph at precisely the carbon
                    # anchor (neighboring labels may already overlap it).
                    if not 1<=len(new)-len(old)<=4:raise
                    remaining=[g['bbox']-np.tile(shift,2) for g in new]
                    for g in old:
                        j=next((j for j,b in enumerate(remaining) if np.allclose(g['bbox'],b,rtol=0,atol=.03)),None)
                        if j is not None:remaining.pop(j)
                    if not any(contains(b) for b in remaining):raise
                    self.hidden_carbon_cache[i]=True
            if plain and self.hidden_carbon_cache[i]:hidden.append(i)
        for obj in self.atoms+self.bonds:obj.unhighlight()
        for i in atom_ids-set(hidden):self.atoms[i].highlight()
        for i in bond_ids:self.bonds[i].highlight()
        assert [a.index() for a in self.atoms if a.isHighlighted()]==sorted(atom_ids-set(hidden))
        assert [b.index() for b in self.bonds if b.isHighlighted()]==sorted(bond_ids)
        renders=[];geometries=[]
        for rgb in COLORS:
            self.ig.setOption('render-highlight-color',','.join(map(str,rgb)))
            svg=bytes(self.renderer.renderToBuffer(self.mol)).decode();self.render_calls+=1
            _,candidate_geometry=geometry_check(self.baseline,svg,(self.glyphs,self.paths));renders.append(svg);geometries.append(candidate_geometry)
            if not np.allclose(self.coords,np.array([a.coords() for a in self.atoms]),rtol=0,atol=1e-6):raise ValueError('Highlight changed atom anchors')
        (ga,pa),(gb,pb)=geometries
        glyphs=[i for i,(a,b) in enumerate(zip(ga,gb)) if a.get('fill')!=b.get('fill')]
        paths=[i for i,(a,b) in enumerate(zip(pa,pb)) if any(a['style'].get(k)!=b['style'].get(k) for k in ('fill','stroke'))]
        result={'atom_ids':sorted(atom_ids),'bond_ids':sorted(bond_ids),'glyph_indices':glyphs,'path_indices':paths,'geometry_verified':True,'unlabeled_vertex_ids':hidden}
        if keep_svg:result['probe_svgs']=renders
        return result


def infer_ownership(ig,renderer,mol,baseline_svg):
    glyph_owners={};path_owners={};atom_reports=[];bond_reports=[]
    with OwnershipProbe(ig,renderer,mol,baseline_svg) as probe:
        for atom in probe.atoms:
            try:hit=probe.select(atom_ids=[atom.index()])
            except ValueError as exc:raise ValueError(f'Atom {atom.index()} ({atom.symbol()}) ownership probe: {exc}') from exc
            atom_reports.append(hit)
            for j in hit['glyph_indices']:
                if j in glyph_owners:raise ValueError('A glyph is owned by multiple native atoms')
                glyph_owners[j]=atom.index()
        for bond in probe.bonds:
            try:hit=probe.select(bond_ids=[bond.index()])
            except ValueError as exc:raise ValueError(f'Bond {bond.index()} ownership probe: {exc}') from exc
            bond_reports.append(hit)
            if hit['glyph_indices']:raise ValueError('Bond highlighting unexpectedly colored an atom label')
            if not hit['path_indices']:raise ValueError(f'No native highlighted ink for bond {bond.index()}')
            for j in hit['path_indices']:
                if j in path_owners:raise ValueError('A path is owned by multiple native bonds')
                path_owners[j]=bond.index()
        calls=probe.render_calls
    # Re-render after restoration to ensure probe-only state cannot leak out.
    restored=bytes(renderer.renderToBuffer(mol)).decode();geometry_check(baseline_svg,restored)
    ag,ap=svg_geometry(baseline_svg);bg,bp=svg_geometry(restored)
    if any(a.get('fill')!=b.get('fill') for a,b in zip(ag,bg)) or any(a['style']!=b['style'] for a,b in zip(ap,bp)):raise ValueError('Highlight probe failed to restore production colors')
    return glyph_owners,path_owners,{'method':'exclusive_native_dual_color_v1','probe_colors':COLORS,'probe_render_calls':calls,'geometry_verified':True,'production_colors_restored':True,'atoms':atom_reports,'bonds':bond_reports}


def glyph_mask(svg,glyph_indices,affine,size):
    """Rasterize only owned glyphs at the exact production affine, no neighbors."""
    glyphs,_=svg_geometry(svg);selected={glyphs[i]['sequence'] for i in glyph_indices}
    tree=ET.fromstring(svg);sequence=0
    def walk(el):
        nonlocal sequence
        if el.tag.endswith('defs'):return
        current=sequence;sequence+=1
        for child in list(el):
            if child.tag.endswith('defs'):continue
            before=sequence;walk(child)
            if child.tag.endswith('path') or child.tag.endswith('rect') or (child.tag.endswith('use') and before not in selected):el.remove(child)
            elif child.tag.endswith('use'):child.set('style','fill:black;fill-opacity:1;stroke:none')
    walk(tree)
    w,h=size;out=ET.Element('{http://www.w3.org/2000/svg}svg',{'width':str(w),'height':str(h),'viewBox':f'0 0 {w} {h}'})
    defs=tree.find('s:defs',NS)
    if defs is not None:out.append(copy.deepcopy(defs))
    m=np.asarray(affine);g=ET.SubElement(out,'{http://www.w3.org/2000/svg}g',{'transform':f'matrix({m[0,0]},{m[1,0]},{m[0,1]},{m[1,1]},{m[0,2]},{m[1,2]})'})
    for child in tree:
        if child is not defs:g.append(child)
    rgba=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ET.tostring(out)))).convert('RGBA'));mask=rgba[:,:,3]>0
    ys,xs=np.nonzero(mask)
    if not len(xs):return np.zeros((0,0),bool),None
    bb=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    return mask[bb[1]:bb[3],bb[0]:bb[2]],bb


def annotate_svg(svg,glyph_owners,path_owners,atom_path_owners=None):
    """Persist ownership so later affine/charge augmentation never guesses again."""
    glyphs,paths=svg_geometry(svg)
    glyph_ids={glyphs[j]['sequence']:owner for j,owner in glyph_owners.items()}
    path_ids={paths[j]['sequence']:owner for j,owner in path_owners.items()}
    atom_path_ids={paths[j]['sequence']:owner for j,owner in (atom_path_owners or {}).items()}
    root=ET.fromstring(svg);sequence=0
    def walk(el):
        nonlocal sequence
        if el.tag.endswith('defs'):return
        i=sequence;sequence+=1
        if i in glyph_ids:el.set('data-owner-atom-id',str(glyph_ids[i]))
        if i in atom_path_ids:el.set('data-owner-atom-id',str(atom_path_ids[i]))
        if i in path_ids:el.set('data-owner-bond-id',str(path_ids[i]))
        for child in el:walk(child)
    walk(root)
    return ET.tostring(root,encoding='unicode')
