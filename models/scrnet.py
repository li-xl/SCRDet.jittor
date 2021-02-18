import jittor as jt 
from jittor import nn 

from models.resnet import resnet50
from models.fpn import SCRDetFPN
from utils import anchor_utils



class SCRNet(nn.Module):

    def __init__(self,cfgs):
        super(SCRNet,self).__init__()

        if cfgs.ANCHOR_MODE == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.cfgs = cfgs

        self.backbone = resnet50(pretrained=True)
        self.fpn = SCRDetFPN(anchor_stride=8,out_dim=512,ratio=16)
        self.rpn = RPN(self.num_anchors_per_location)
    
    def make_anchors(self, feature_to_cropped):
        featuremap_height, featuremap_width =feature_to_cropped.shape[-2], feature_to_cropped.shape[-1]

        anchors = anchor_utils.make_anchors(base_anchor_size=self.cfgs.BASE_ANCHOR_SIZE_LIST,
                                            anchor_scales=self.cfgs.ANCHOR_SCALES, anchor_ratios=self.cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=self.cfgs.ANCHOR_STRIDE)
        return anchors

    def execute(self,x):
        img_shape = x.shape
        # backbone
        C3,C4 = self.backbone(x)
        feature,pa_mask = self.fpn(C3,C4)

        # rpn
        rpn_box_pred, rpn_cls_score = self.rpn(feature)
        rpn_box_pred = jt.reshape(rpn_box_pred, [-1, 4])
        rpn_cls_score = jt.reshape(rpn_cls_score, [-1, 2])
        rpn_cls_prob = nn.softmax(rpn_cls_score)

        # anchors
        anchors = self.make_anchors(feature)

        # postprocess rpn proposals
        rois, roi_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,rpn_cls_prob=rpn_cls_prob,img_shape=img_shape,anchors=anchors)

        
        return features


