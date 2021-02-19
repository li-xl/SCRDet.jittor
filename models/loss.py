import jittor as jt
from utils.box_utils import rbbox_transform_inv

def smooth_l1_loss_base(self, bbox_pred, bbox_targets, sigma=1.0):
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


def smooth_l1_loss_rcnn(self, bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
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
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = bbox_pred.shape[0]
    bbox_loss = tf.reduce_sum(
            tf.reduce_sum(value * inside_mask, 1) * outside_mask) / normalizer

    return bbox_loss


def iou_smooth_l1_loss_rcnn_r(self, bbox_pred, bbox_targets, label, rois, target_gt_r, num_classes, sigma=1.0):
    '''
    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 5]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 5]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = (label>0).float32()

    target_gt_r = tf.reshape(tf.tile(tf.reshape(target_gt_r, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])
    x_c = (rois[:, 2] + rois[:, 0]) / 2
    y_c = (rois[:, 3] + rois[:, 1]) / 2
    h = rois[:, 2] - rois[:, 0] + 1
    w = rois[:, 3] - rois[:, 1] + 1
    theta = -90 * tf.ones_like(x_c)
    rois = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
    rois = tf.reshape(tf.tile(tf.reshape(rois, [-1, 1, 5]), [1, num_classes, 1]), [-1, 5])

    boxes_pred = rbbox_transform_inv(boxes=rois, deltas=tf.reshape(bbox_pred, [-1, 5]),
                                                        scale_factors=self.cfgs.ROI_SCALE_FACTORS)
    overlaps = tf.py_func(iou_rotate_calculate2,
                              inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_gt_r, [-1, 5])],
                              Tout=[tf.float32])
    overlaps = tf.reshape(overlaps, [-1, num_classes])

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 5])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 5])

    value = smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                                 depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
            tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    iou_factor = tf.stop_gradient(tf.exp((1 - overlaps)) - 1) / (tf.stop_gradient(value) + self.cfgs.EPSILON)

    regression_loss = tf.reduce_sum(value * inside_mask * iou_factor, 1)

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(regression_loss * outside_mask) / normalizer

    return bbox_loss


def attention_loss(self, mask, featuremap):
    # shape = mask.get_shape().as_list()
    shape = tf.shape(mask)
    featuremap = tf.image.resize_bilinear(featuremap, [shape[0], shape[1]])

    mask = tf.cast(mask, tf.int32)
    mask = tf.reshape(mask, [-1, ])
    featuremap = tf.reshape(featuremap, [-1, 2])
    attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask, logits=featuremap)
    attention_loss = tf.reduce_mean(attention_loss)
    return attention_loss