import jittor as jt
from utils.box_utils import loc2bbox_r
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
    loss_box = (box_diff.sqr() * (sigma_2 / 2.0) * smoothL1_sign) + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''
    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''
    
    value = smooth_l1_loss_base(bbox_pred,bbox_targets,sigma=sigma)
    value = value.sum(dim=1)
    value = value*((label>0).float32().stop_grad())
    # bbox_loss = value.sum() / bbox_pred.shape[0]
    bbox_loss = value.sum()/jt.maximum((label!=-1).float32().sum().stop_grad(),1)
    return bbox_loss

def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, sigma=1.0):
    value = smooth_l1_loss_base(bbox_pred,bbox_targets,sigma=sigma)
    value = value.sum(dim=1)
    value = value*((label>0).float32().stop_grad())
    # bbox_loss = value.sum() / bbox_pred.shape[0]
    bbox_loss = value.mean()
    return bbox_loss

def iou_smooth_l1_loss_rcnn_r(bbox_pred, bbox_targets, label,rois,sigma=1.0):

    outside_mask = (label>0).float32().stop_grad()

    boxes_pred = loc2bbox_r(rois,bbox_pred)
    target_gt_r = loc2bbox_r(rois,bbox_targets)
    overlaps = iou_rotate(boxes_pred,target_gt_r)

    value = smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = value.sum(1)
    iou_factor = ((jt.exp(1 - overlaps) - 1) / (value + 1e-5)).stop_grad()
    value = value*iou_factor
    value = value*outside_mask
    # bbox_loss = value.sum() / bbox_pred.shape[0]
    bbox_loss = value.mean()
    return bbox_loss


def attention_loss(mask, featuremap):
    # featuremap:[n,c,h,w]
    # mask: [n,c,h,w]
    assert mask.ndim==4 and featuremap.ndim==4
    featuremap = jt.nn.interpolate(featuremap, [mask.shape[-2],mask.shape[-1]])
    mask = mask.transpose(0,2,3,1).reshape([-1, ]).int32()
    featuremap = featuremap.transpose(0,2,3,1).reshape([-1, 2])
    attention_loss = jt.nn.cross_entropy_loss(featuremap,mask)
    return attention_loss