import jittor as jt 
import numpy as np 

def loc2bbox(src_bbox,loc):
    if src_bbox.shape[0] == 0:
        return jt.zeros((0, 4), dtype=loc.dtype)

    src_width = src_bbox[:, 2:3] - src_bbox[:, 0:1]
    src_height = src_bbox[:, 3:4] - src_bbox[:, 1:2]
    src_center_x = src_bbox[:, 0:1] + 0.5 * src_width
    src_center_y = src_bbox[:, 1:2] + 0.5 * src_height

    dx = loc[:, 0:1]
    dy = loc[:, 1:2]
    dw = loc[:, 2:3]
    dh = loc[:, 3:4]

    center_x = dx*src_width+src_center_x
    center_y = dy*src_height+src_center_y
        
    w = jt.exp(dw.minimum(20.0)) * src_width
    h = jt.exp(dh.minimum(20.0)) * src_height
        
    x1,y1,x2,y2 = center_x-0.5*w, center_y-0.5*h, center_x+0.5*w, center_y+0.5*h
        
    dst_bbox = jt.contrib.concat([x1,y1,x2,y2],dim=1)

    return dst_bbox
    
def bbox2loc(src_bbox,dst_bbox):        
    width = src_bbox[:, 2:3] - src_bbox[:, 0:1]
    height = src_bbox[:, 3:4] - src_bbox[:, 1:2]
    center_x = src_bbox[:, 0:1] + 0.5 * width
    center_y = src_bbox[:, 1:2] + 0.5 * height

    base_width = dst_bbox[:, 2:3] - dst_bbox[:, 0:1]
    base_height = dst_bbox[:, 3:4] - dst_bbox[:, 1:2]
    base_center_x = dst_bbox[:, 0:1] + 0.5 * base_width
    base_center_y = dst_bbox[:, 1:2] + 0.5 * base_height

    eps = 1e-5
    height = jt.maximum(height, eps)
    width = jt.maximum(width, eps)

    dy = (base_center_y - center_y) / height
    dx = (base_center_x - center_x) / width

    dw = jt.log(base_width / width)
    dh = jt.log(base_height / height)
        
    loc = jt.contrib.concat([dx,dy,dw,dh],dim=1)
    return loc
    