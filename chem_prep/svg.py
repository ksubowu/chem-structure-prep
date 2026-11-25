"""Visible Indigo SVG ink -> charge boxes, cropped instance masks and band quads.

The ordered PCA/taper quad convention follows band_training/prepare.py:region_quad
in portable_mp123_human_band_transfer_20261110. Indigo SVG has no bond IDs;
new renders use exclusive native highlighting; geometric assignment remains for legacy audits.
"""
import math,re,xml.etree.ElementTree as ET
import cairocffi as cairo
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.svgLib.path import parse_path
NS={'s':'http://www.w3.org/2000/svg'}

class PointsPen(BasePen):
    def __init__(self):super().__init__(None);self.points=[]
    def _moveTo(self,p):self.points.append(p)
    def _lineTo(self,p):self.points.append(p)
    def _curveToOne(self,a,b,c):
        p=np.array(self._getCurrentPoint());a,b,c=map(np.array,(a,b,c))
        for t in np.linspace(0,1,17)[1:]:self.points.append((1-t)**3*p+3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3*c)
    def _qCurveToOne(self,a,b):
        p=np.array(self._getCurrentPoint());a,b=map(np.array,(a,b))
        for t in np.linspace(0,1,17)[1:]:self.points.append((1-t)**2*p+2*(1-t)*t*a+t*t*b)
    def _closePath(self):pass
    def _endPath(self):pass

def matrix(value):
    if not value:return np.eye(3)
    m=re.fullmatch(r'matrix\(([^)]+)\)',value.strip())
    if not m:raise ValueError('Unsupported Indigo SVG transform '+value)
    a,b,c,d,e,f=map(float,re.split(r'[,\s]+',m[1].strip()))
    return np.array([[a,c,e],[b,d,f],[0,0,1.]])

def transformed(points,mat):
    return np.column_stack((points,np.ones(len(points))))@mat[:2,:].T

def style(s):return dict(x.strip().split(':',1) for x in s.split(';') if ':' in x)

def svg_geometry(svg):
    root=ET.fromstring(svg);defs={};glyphs=[];paths=[];sequence=0
    for symbol in root.findall('.//s:symbol',NS):
        pen=BoundsPen(None)
        for p in symbol.findall('s:path',NS):parse_path(p.get('d'),pen)
        defs[symbol.get('id')]=pen.bounds
    def walk(el,mat,inherited):
        nonlocal sequence
        if el.tag.endswith('defs'):return
        mat=mat@matrix(el.get('transform'));sty={**inherited,**style(el.get('style',''))}
        if el.tag.endswith('use'):
            key=el.get('{http://www.w3.org/1999/xlink}href','').lstrip('#');bounds=defs.get(key)
            if bounds:
                a,b,c,d=bounds;x=float(el.get('x','0'));y=float(el.get('y','0'))
                p=transformed(np.array([[a+x,b+y],[c+x,b+y],[c+x,d+y],[a+x,d+y]]),mat)
                glyphs.append({'bbox':np.r_[p.min(0),p.max(0)],'sequence':sequence,'key':key,'baseline':y,'fill':sty.get('fill'),'native_atom_id':int(el.get('data-owner-atom-id')) if el.get('data-owner-atom-id') is not None else None})
        elif el.tag.endswith('path'):
            pen=PointsPen();parse_path(el.get('d'),pen)
            if pen.points:
                pts=transformed(np.array(pen.points),mat)
                paths.append({'d':el.get('d'),'matrix':mat,'style':sty,'points':pts,'bbox':np.r_[pts.min(0),pts.max(0)],'sequence':sequence,'native_atom_id':int(el.get('data-owner-atom-id')) if el.get('data-owner-atom-id') is not None else None,'native_bond_id':int(el.get('data-owner-bond-id')) if el.get('data-owner-bond-id') is not None else None})
        sequence+=1
        for child in el:walk(child,mat,sty)
    walk(root,np.eye(3),{})
    return glyphs,paths

def charge_and_atom_glyphs(glyphs,paths,coords,formal_charges,glyph_owners=None,atom_path_owners=None):
    charged=[i for i,q in enumerate(formal_charges) if q]
    candidates=[]
    for j,p in enumerate(paths):
        # Cairo draws +/- as absolute filled polygons; bond strokes/wedges use
        # the molecule-coordinate matrix. Reject round radical markers.
        if not np.allclose(p['matrix'],np.eye(3)) or p['style'].get('stroke','none')!='none':continue
        points=p['points'];a,b,c,d=p['bbox'];w,h=c-a,d-b
        if min(w,h)<=0:continue
        if len(points) in (4,5) and w/h>2 and cv2.contourArea(cv2.convexHull(points.astype(np.float32)))/(w*h)>.95:candidates.append((j,-1))
        elif len(points) in (12,13) and .65<w/h<1.5:candidates.append((j,1))
    if len(candidates)<len(charged):raise ValueError(f'Missing charge SVG signs: {len(candidates)}/{len(charged)}')
    charges=[];excluded=set();used_paths=set()
    if charged:
        costs=np.full((len(charged),len(candidates)),1e6)
        for ri,i in enumerate(charged):
            for cj,(j,sign) in enumerate(candidates):
                p=paths[j];center=(p['bbox'][:2]+p['bbox'][2:])/2
                if sign==np.sign(formal_charges[i]) and (atom_path_owners is None or atom_path_owners.get(j)==i):costs[ri,cj]=np.linalg.norm(center-coords[i])
        rows,cols=linear_sum_assignment(costs)
        for ri,cj in zip(rows,cols):
            if costs[ri,cj]>=1e6:raise ValueError('Charge sign mismatch')
            i=charged[ri];j,_=candidates[cj];p=paths[j];boxes=[p['bbox']];used_paths.add(j)
            if abs(formal_charges[i])>1:
                # Magnitude glyph immediately precedes its absolute sign path.
                before=[(k,g) for k,g in enumerate(glyphs) if g['sequence']<p['sequence'] and k not in excluded and (glyph_owners is None or glyph_owners.get(k)==i)]
                before.sort(key=lambda x:x[1]['sequence'],reverse=True)
                count=len(str(abs(formal_charges[i])))
                for k,g in before[:count]:
                    center=(g['bbox'][:2]+g['bbox'][2:])/2
                    distance=np.linalg.norm(center-(p['bbox'][:2]+p['bbox'][2:])/2)
                    if distance>30:raise ValueError('Missing nearby charge magnitude glyph')
                    excluded.add(k);boxes.append(g['bbox'])
            boxes=np.array(boxes);bb=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]
            charges.append({'atom_id':i,'value':int(formal_charges[i]),'label':f'{formal_charges[i]:+d}','raw_bbox':bb})
    grouped={i:[] for i in range(len(coords))}
    for j,g in enumerate(glyphs):
        if j in excluded:continue
        center=(g['bbox'][:2]+g['bbox'][2:])/2
        if glyph_owners is not None and j not in glyph_owners:raise ValueError(f'Glyph {j} has no native atom owner')
        i=glyph_owners[j] if glyph_owners is not None else int(np.argmin(np.linalg.norm(coords-center,axis=1)))
        grouped[i].append(g['bbox'])
    return grouped,charges,used_paths

def assign_bond_paths(paths,coords,bonds,exclude,*,_angular_weight=1.5):
    groups=[[] for _ in bonds]
    if not bonds:return groups,[]
    starts=np.array([coords[b.source().index()] for b in bonds]);ends=np.array([coords[b.destination().index()] for b in bonds])
    delta=ends-starts;length=np.linalg.norm(delta,axis=1);axis=delta/np.maximum(length[:,None],1e-8);normal=np.stack((-axis[:,1],axis[:,0]),axis=1)
    unassigned=[];eligible=[];all_costs=[]
    for j,p in enumerate(paths):
        if j in exclude:continue
        if np.allclose(p['matrix'],np.eye(3)) and np.max(p['bbox'][2:]-p['bbox'][:2])<.2*np.median(length):
            # Tiny absolute dots are radical/label marks. Solid wedges are also
            # absolute filled paths, but extend along a visible bond.
            unassigned.append(j);continue
        pts=p['points'];rel=pts[None,:,:]-starts[:,None,:]
        t=np.sum(rel*axis[:,None,:],axis=2)/length[:,None]
        u=np.abs(np.sum(rel*normal[:,None,:],axis=2))/length[:,None]
        cost=np.mean(u,axis=1)+.2*np.abs(t.mean(1)-.5)+2*np.mean(np.maximum(-t,0)+np.maximum(t-1,0),axis=1)
        # A whole hashed path is assigned together. Short transverse dash strokes
        # must not be mistaken for their neighbors' central axes.
        centered=pts-pts.mean(0)
        if len(pts)>1:
            _,vecs=np.linalg.eigh(centered.T@centered);direction=vecs[:,-1]
            span=np.ptp(centered@direction)
            if span>.05*np.median(length):
                alignment=np.clip(np.abs(axis@direction),0,1)
                cost+=.65*(1-alignment)
                # 1-cos(theta) is quadratic near zero: it barely distinguishes
                # a carbonyl stroke from a crossing bridge-ring edge 20 degrees
                # away. A straight stroke needs a first-order angular penalty.
                # Keep wedge/hashed compounds on the original PCA heuristic;
                # their outline segments need not follow the bond axis.
                residual=np.linalg.norm(centered-(centered@direction)[:,None]*direction)
                if residual<1e-5*max(span,1):
                    cost+=_angular_weight*np.sqrt(np.maximum(0,1-alignment**2))
        eligible.append(p);all_costs.append(cost)
    if not eligible:return groups,unassigned
    costs=np.array(all_costs)
    owners=costs.argmin(1)
    if len(set(owners))<len(bonds):
        # Resolve competition near shared vertices globally. Each drawn bond
        # needs a distinct seed primitive; parallel strokes remain extra paths.
        if len(eligible)<len(bonds):raise ValueError('Fewer visible SVG primitives than graph bonds')
        rows,cols=linear_sum_assignment(costs.T)
        for bidx,pidx in zip(rows,cols):owners[pidx]=bidx
    for p,cost,idx in zip(eligible,costs,owners):
        p['assignment_cost']=float(cost[idx]);groups[idx].append(p)
    return groups,unassigned

def raster_paths(paths,affine,size):
    w,h=size
    pts=np.concatenate([transformed(p['points'],affine) for p in paths])
    widths=[]
    for p in paths:
        m=(affine@p['matrix'])[:2,:2]
        widths.append(float(p['style'].get('stroke-width',0))*np.linalg.norm(m,ord=2))
    pad=max(widths+[1])/2+2
    lo=np.maximum(0,np.floor(pts.min(0)-pad)).astype(int);hi=np.minimum([w,h],np.ceil(pts.max(0)+pad+1)).astype(int)
    ww,hh=hi-lo
    if min(ww,hh)<=0:raise ValueError('Empty bond raster bounds')
    surface=cairo.ImageSurface(cairo.FORMAT_A8,int(ww),int(hh));ctx=cairo.Context(surface)
    for p in paths:
        ctx.save();m=affine@p['matrix'];m=m.copy();m[:2,2]-=lo
        ctx.transform(cairo.Matrix(m[0,0],m[1,0],m[0,1],m[1,1],m[0,2],m[1,2]))
        parse_path(p['d'],CairoPen(None,ctx));sty=p['style'];ctx.set_source_rgba(1,1,1,1)
        if sty.get('fill','black')!='none':ctx.fill_preserve()
        if sty.get('stroke','none')!='none':
            ctx.set_line_width(float(sty.get('stroke-width','1')))
            ctx.set_line_cap({'butt':0,'round':1,'square':2}.get(sty.get('stroke-linecap'),0))
            ctx.set_line_join({'miter':0,'round':1,'bevel':2}.get(sty.get('stroke-linejoin'),0));ctx.set_miter_limit(float(sty.get('stroke-miterlimit','10')));ctx.stroke()
        else:ctx.new_path()
        ctx.restore()
    surface.flush();data=np.frombuffer(surface.get_data(),np.uint8).reshape(int(hh),surface.get_stride())[:,:int(ww)].copy()
    ink=data>0
    if not ink.any():raise ValueError('Empty actual SVG bond ink')
    return ink,[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]

def region_quad(ink,category,stroke_axis=None):
    """Reference PCA/tapered ordered region, from visible ink only."""
    ys,xs=np.nonzero(ink);pts=np.column_stack((xs,ys)).astype(float);center=pts.mean(0)
    if len(pts)<2:raise ValueError('Insufficient band ink')
    _,vectors=np.linalg.eigh(np.cov((pts-center).T));axis=vectors[:,-1]
    if stroke_axis is not None and abs(np.dot(axis,stroke_axis))<.9:axis=np.asarray(stroke_axis)
    if axis[0]<0 or (abs(axis[0])<1e-7 and axis[1]<0):axis=-axis
    normal=np.array([-axis[1],axis[0]]);t=(pts-center)@axis;u=(pts-center)@normal
    lo,hi=t.min(),t.max();length=hi-lo;widths=[]
    for a,b in [(lo,lo+.3*length),(hi-.3*length,hi)]:
        us=u[(t>=a)&(t<=b)];widths.append(max(1.,float(np.max(np.abs(us)))+.5))
    if category in (5,6) and widths[0]>widths[1]:
        axis=-axis;normal=-normal;lo,hi=-hi,-lo;widths=widths[::-1]
    if category not in (5,6):widths=[max(widths)]*2
    a=center+lo*axis;b=center+hi*axis
    return np.stack((a+widths[0]*normal,a-widths[0]*normal,b-widths[1]*normal,b+widths[1]*normal))

def make_bands(paths,coords,bonds,exclude,affine,size,path_owners=None):
    if path_owners is None:groups,unassigned=assign_bond_paths(paths,coords,bonds,exclude)
    else:
        groups=[[] for _ in bonds];unassigned=[]
        for j,p in enumerate(paths):
            if j in exclude:continue
            if j not in path_owners:unassigned.append(j);continue
            p['assignment_cost']=0.;groups[path_owners[j]].append(p)
    bands=[];masks={}
    w,h=size
    for i,(bond,group) in enumerate(zip(bonds,groups)):
        if not group:raise ValueError(f'No visible SVG primitives assigned to bond {i}')
        ink,bb=raster_paths(group,affine,size);ys,xs=np.nonzero(ink)
        hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32)).reshape(-1,2)
        region=np.zeros(ink.shape,np.uint8);cv2.fillConvexPoly(region,hull,1);region|=ink
        category=bond.bondStereo() if bond.bondStereo() in (5,6) else bond.bondOrder()
        # For a short double/triple bond, the distance between the parallel
        # strokes can exceed their visible length. Plain ink PCA then turns 90
        # degrees. Recover their shared direction from the actual SVG strokes,
        # without using atom anchors to define the band region.
        consensus=np.zeros((2,2));stroke_axis=None
        if category in (1,2,3,4):
            for p in group:
                points=transformed(p['points'],affine);centered=points-points.mean(0)
                eig,vec=np.linalg.eigh(centered.T@centered);direction=vec[:,-1]
                span=np.ptp(centered@direction)
                if span>0:consensus+=span*np.outer(direction,direction)
            eig,vec=np.linalg.eigh(consensus)
            if eig[-1]>0 and eig[-1]>3*eig[0]:stroke_axis=vec[:,-1]
        quad=region_quad(ink,category,stroke_axis)+np.array(bb[:2]);hull=hull+np.array(bb[:2])
        # Pixel coordinates are bounded by the image; end-face ordering survives.
        quad=np.clip(quad,[0,0],[w-1,h-1]);hull=np.clip(hull,[0,0],[w-1,h-1])
        masks[f'ink_{i}']=ink;masks[f'region_{i}']=region.astype(bool)
        bands.append({'id':i,'bond_id':bond.index(),'category_id':category,'quad':np.round(quad,3).tolist(),'region_polygon':hull.tolist(),'mask_bbox':bb,'mask_key':f'ink_{i}','region_mask_key':f'region_{i}','ink_area':int(ink.sum()),'region_area':int(region.sum()),'svg_primitive_count':len(group),'assignment_max_cost':max(p['assignment_cost'] for p in group),'geometry_source':'actual_Indigo_SVG_bond_ink','primitive_assignment':'native_exclusive_highlight' if path_owners is not None else 'geometric_to_rendered_bond_anchors_v2_angle', 'svg_primitive_ids':[p['sequence'] for p in group]})
    return bands,masks,unassigned
