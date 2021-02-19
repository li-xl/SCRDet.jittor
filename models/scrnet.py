import jittor as jt 
from jittor import nn 

from models.resnet import resnet50
from models.fpn import SCRDetFPN
from models.rpn import RegionProposalNetwork,AnchorTargetCreator,ProposalCreator,ProposalTargetCreator
from models.roi_head import RoIHead



class SCRNet(nn.Module):

    def __init__(self,cfgs,n_class):
        super(SCRNet,self).__init__()

        if cfgs.ANCHOR_MODE == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.cfgs = cfgs

        self.backbone = resnet50(pretrained=True)
        self.fpn = SCRDetFPN(anchor_stride=8,out_dim=512,ratio=16)
        self.n_class = n_class

        self.rpn = RegionProposalNetwork(in_channels=self.backbone.out_channels, 
                                        mid_channels=512, 
                                        ratios=[0.5, 1, 2],
                                        anchor_scales=[8, 16, 32], 
                                        feat_stride=self.backbone.feat_stride,
                                        nms_thresh=0.7,
                                        n_train_pre_nms=12000,
                                        n_train_post_nms=2000,
                                        n_test_pre_nms=6000,
                                        n_test_post_nms=300,
                                        min_size=16,)

        self.anchor_target_creator = AnchorTargetCreator(n_sample=256,
                                                         pos_iou_thresh=0.7, 
                                                         neg_iou_thresh=0.3,
                                                         pos_ratio=0.5)

        self.proposal_target_creator = ProposalTargetCreator(n_sample=128,
                                                             pos_ratio=0.25, 
                                                             pos_iou_thresh=0.5,
                                                             neg_iou_thresh_hi=0.5, 
                                                             neg_iou_thresh_lo=0.0)
        
        self.head = RoIHead(in_channels=self.backbone.out_channels,
                            n_class=n_class,
                            roi_size=7,
                            spatial_scale=1.0/self.backbone.feat_stride,
                            sampling_ratio=0)

    def _forward_train(self,x):
        pass 

    def _forward_test(self,x):
        pass 

    def execute(self,x):
        img_shape = x.shape
        # backbone
        C3,C4 = self.backbone(x)
        feature,pa_mask = self.fpn(C3,C4)

        # rpn
        rpn_box_pred, rpn_cls_score = self.rpn(feature)
        


