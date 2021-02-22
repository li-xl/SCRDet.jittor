import jittor as jt 
import numpy as np 
import cv2

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
    
def convert_rotation_box(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

            if theta == 0:
                w, h = h, w
                theta -= 90

            boxes.append([x, y, w, h, theta])
    return np.array(boxes, dtype=np.float32)

def convert_horizen_box(coordinate,with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_min, y_min, x_max, y_max, (label)]
    """
    boxes = []
    if with_label:
        boxes = np.array(coordinate,dtype=np.float32).reshape(-1,9)
        x1, y1, x2, y2, x3, y3, x4, y4, label = np.split(boxes,9,axis=1)
    else:
        boxes = np.array(coordinate,dtype=np.float32).reshape(-1,8)
        x1, y1, x2, y2, x3, y3, x4, y4 = np.split(boxes,8,axis=1)

    x_min = np.minimum(np.minimum(x1,x2),np.minimum(x3,x4))
    x_max = np.maximum(np.maximum(x1,x2),np.maximum(x3,x4))
    y_min = np.minimum(np.minimum(y1,y2),np.minimum(y3,y4))
    y_max = np.maximum(np.maximum(y1,y2),np.maximum(y3,y4))

    if with_label:
        return np.concatenate([x_min,y_min,x_max,y_max,label],axis=1)
    else:
        return np.concatenate([x_min,y_min,x_max,y_max],axis=1)


def convert_coordinate(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)

def rbbox_transform_inv(boxes, deltas, scale_factors=None):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dtheta = deltas[:, 4]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]
        dtheta /= scale_factors[4]

    # BBOX_XFORM_CLIP = tf.log(cfgs.IMG_SHORT_SIDE_LEN / 16.)
    # dw = tf.minimum(dw, BBOX_XFORM_CLIP)
    # dh = tf.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * boxes[:, 2] + boxes[:, 0]
    pred_ctr_y = dy * boxes[:, 3] + boxes[:, 1]
    pred_w = jt.exp(dw) * boxes[:, 2]
    pred_h = jt.exp(dh) * boxes[:, 3]

    pred_theta = dtheta * 180 / np.pi + boxes[:, 4]

    return jt.transpose(jt.stack([pred_ctr_x, pred_ctr_y,
                                  pred_w, pred_h, pred_theta]),[1,0])