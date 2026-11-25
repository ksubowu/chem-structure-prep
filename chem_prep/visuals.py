"""Raster statistics and human-review overlays."""
import cv2
import numpy as np
from PIL import Image,ImageDraw

def image_stats(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys,xs=np.where(mask>0)
    if not len(xs): raise ValueError('blank image')
    bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    # Stroke estimate from the ridges of the foreground distance transform.
    dist=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    ridge=(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8))-0.01)&(dist>0)
    stroke=float(np.median(dist[ridge])*2)
    return bbox,stroke

def overlay(image, ann, masks=None):
    im=Image.fromarray(image).convert('RGBA');layer=Image.new('RGBA',im.size,(0,0,0,0));paint=ImageDraw.Draw(layer)
    w,h=im.size;lw=max(1,round(min(w,h)/350))
    for band in ann['bands']:
        color=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(band['id']*137.5)%180,210,210]]]),cv2.COLOR_HSV2RGB)[0,0])
        polygon=[tuple(p) for p in band['quad']];paint.polygon(polygon,fill=color+(48,),outline=color+(210,),width=lw)
    im=Image.alpha_composite(im,layer).convert('RGB')
    if masks is not None:
        pix=np.array(im)
        for band in ann['bands']:
            x1,y1,x2,y2=band['mask_bbox'];ink=masks[band['mask_key']]
            color=cv2.cvtColor(np.uint8([[[int(band['id']*137.5)%180,210,210]]]),cv2.COLOR_HSV2RGB)[0,0]
            roi=pix[y1:y2,x1:x2];roi[ink]=(.4*roi[ink]+.6*color).astype(np.uint8)
        im=Image.fromarray(pix)
    draw=ImageDraw.Draw(im)
    for a in ann['atoms']:
        box=a['bbox'];draw.rectangle(box,outline=(30,100,215) if a['visible_label'] else (20,160,75),width=lw)
        draw.text((box[0],max(0,box[1]-12)),str(a['id']),fill=(100,0,145))
    for charge in ann['charges']:
        box=charge['bbox'];draw.rectangle(box,outline=(240,20,115),width=lw)
        draw.text((box[2]+1,box[1]),charge['label'],fill=(220,0,100))
    return im


def review(reference,rendered,marked,title):
    items=[(Image.fromarray(rendered),'Rendered'),(marked,'Atom / charge boxes + bond bands')]
    if reference is not None:items.insert(0,(Image.fromarray(reference),'Source reference'))
    canvas=Image.new('RGB',(620*len(items),652),'#eeeeee')
    for i,(im,label) in enumerate(items):
        im=im.copy();im.thumbnail((600,560),Image.Resampling.LANCZOS)
        panel=Image.new('RGB',(620,610),'white');panel.paste(im,((620-im.width)//2,40+(560-im.height)//2))
        ImageDraw.Draw(panel).text((10,12),label,fill='black');canvas.paste(panel,(i*620,32))
    draw=ImageDraw.Draw(canvas);draw.text((10,10),title,fill='black')
    draw.text((10,640),'Blue: atom labels | Pink: charges | Green: unlabeled vertices | Colored regions: actual bond ink/bands',fill='black')
    return canvas
