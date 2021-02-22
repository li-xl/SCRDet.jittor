import jittor as jt
from utils.box_utils import rbbox_transform_inv
from utils.iou_rotate import iou_rotate

def smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''
    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    abs_box_diff = jt.abs(box_diff)
    smoothL1_sign = (abs_box_diff<1. / sigma_2).float32()
    loss_box = box_diff.sqr() * (sigma_2 / 2.0) * smoothL1_sign + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''
    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = (label>0).float32()

    value = smooth_l1_loss_base(bbox_pred,bbox_targets,sigma=sigma)
    value = value.sum(dim=2)
    value = value*outside_mask.unsqueeze(1)
    value = value[jt.index((label.shape[0],),dim=0),jt.maximum(label,0)]
    bbox_loss = value.sum() / bbox_pred.shape[0]
    return bbox_loss


def iou_smooth_l1_loss_rcnn_r(bbox_pred, bbox_targets, label, rois, target_gt_r, num_classes, roi_scale_factor,epsilon=1e-5,sigma=1.0):

    outside_mask = (label>0).float32()

    target_gt_r = jt.reshape(jt.repeat(jt.reshape(target_gt_r, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])
    x_c = (rois[:, 2:3] + rois[:, 0:1]) / 2
    y_c = (rois[:, 3:4] + rois[:, 1:2]) / 2
    h = rois[:, 2:3] - rois[:, 0:1] + 1
    w = rois[:, 3:4] - rois[:, 1:2] + 1
    theta = -90 * jt.ones_like(x_c)
    rois = jt.contrib.concat([x_c, y_c, w, h, theta],dim=1)
    rois = jt.reshape(jt.repeat(jt.reshape(rois, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])

    boxes_pred = rbbox_transform_inv(boxes=rois, deltas=jt.reshape(bbox_pred, [-1, 5]),scale_factors=roi_scale_factor)
    overlaps = iou_rotate(boxes_pred,target_gt_r)
    overlaps = jt.reshape(overlaps, [-1, num_classes])

    bbox_pred = jt.reshape(bbox_pred, [-1, num_classes, 5])
    bbox_targets = jt.reshape(bbox_targets, [-1, num_classes, 5])

    value = smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = value.sum(2)
    iou_factor = ((jt.exp(1 - overlaps) - 1) / (value + epsilon)).stop_grad()

    value = value*iou_factor
    value = value*outside_mask.unsqueeze(1)
    regression_loss = value[jt.index((label.shape[0],),dim=0),jt.maximum(label,0)]
    bbox_loss = regression_loss / bbox_pred.shape[0]

    return bbox_loss


def attention_loss(mask, featuremap):
    # featuremap:[n,c,h,w]
    # mask: [n,c,h,w]
    featuremap = jt.nn.interpolate(featuremap, [mask.shape[-2],mask.shape[-1]])
    mask = mask.reshape([-1, ]).int32()
    featuremap = featuremap.reshape([-1, 2])
    featuremap = featuremap.softmax(1)
    attention_loss = jt.nn.cross_entropy_loss(featuremap,mask)
    return attention_loss