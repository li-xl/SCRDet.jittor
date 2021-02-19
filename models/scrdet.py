import jittor as jt 
from jittor import nn 

from models.resnet import resnet50
from models.fpn import SCRDetFPN
from models.rpn import RegionProposalNetwork,AnchorTargetCreator,ProposalCreator,ProposalTargetCreator
from models.roi_head import RoIHead



class SCRDet(nn.Module):

    def __init__(self,n_class):
        super(SCRDet,self).__init__()

        anchor_scales = []
        anchor_ratios = []
        anchor_angles = []
        anchor_mode = 'H'

        if anchor_mode == 'H':
            self.num_anchors_per_location = len(anchor_scales) * len(anchor_ratios)
        else:
            self.num_anchors_per_location = len(anchor_scales) * len(anchor_ratios) * len(anchor_angles)
        self.cfgs = cfgs

        self.backbone = resnet50(pretrained=True)
        self.fpn = SCRDetFPN(anchor_stride=8,out_dim=512,ratio=16)
        self.n_class = n_class

        self.rpn = RegionProposalNetwork(in_channels=512, 
                                        mid_channels=512, 
                                        ratios=[0.5, 1, 2],
                                        anchor_scales=[8, 16, 32], 
                                        feat_stride=1,
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
        
        self.head = RoIHead(in_channels=512,
                            n_class=n_class,
                            roi_size=7,
                            spatial_scale=1.0,
                            sampling_ratio=0)

    def _forward_train(self,feature,pa_mask,img_sizes,hbb,rbb,labels):
        N = feature.shape[0]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(feature, img_sizes)
        
        sample_rois = []
        gt_roi_locs = []
        gt_roi_labels = []
        sample_roi_indexs = []
        gt_rpn_locs = []
        gt_rpn_labels = []
        for i in range(N):
            index = jt.where(roi_indices == i)[0]
            roi = rois[index,:]
            box = hbb[i]
            label = labels[i]
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi,box,label)
            sample_roi_index = i*jt.ones((sample_roi.shape[0],))
            
            sample_rois.append(sample_roi)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_locs.append(gt_roi_loc)
            sample_roi_indexs.append(sample_roi_index)
            
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(box,anchor,img_sizes)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            
        sample_roi_indexs = jt.contrib.concat(sample_roi_indexs,dim=0)
        sample_rois = jt.contrib.concat(sample_rois,dim=0)

        bbox_pred_h,cls_score_h,bbox_pred_r,cls_score_r = self.head(feature,sample_rois,sample_roi_indexs)
        
        # ------------------ RPN losses -------------------#
        rpn_locs = rpn_locs.reshape(-1,4)
        rpn_scores = rpn_scores.reshape(-1,2)
        gt_rpn_labels = jt.contrib.concat(gt_rpn_labels,dim=0)
        gt_rpn_locs = jt.contrib.concat(gt_rpn_locs,dim=0)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_locs,gt_rpn_locs,gt_rpn_labels,self.rpn_sigma)
        rpn_cls_loss = nn.cross_entropy_loss(rpn_scores[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
        gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)
        n_sample = bbox_pred_h.shape[0]
        roi_cls_loc = bbox_pred_h.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[jt.arange(0, n_sample).int32(), gt_roi_labels]
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc,gt_roi_locs,gt_roi_labels,self.roi_sigma)
        roi_cls_loss = nn.cross_entropy_loss(cls_score_h, gt_roi_labels)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]
        return losses
    
    def _forward_test(self,features,img_size):
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size)
        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)
        return rpn_locs, rpn_scores,roi_cls_locs, roi_scores, rois, roi_indices

    def execute(self,batch_imgs,img_sizes,hbb=None,rbb=None,labels=None):
        # backbone
        C3,C4 = self.backbone(batch_imgs)
        feature,pa_mask = self.fpn(C3,C4)

        if self.is_training():
            assert hbb is not None and rbb is not None and labels is not None, "Model must has ground truth"
            return self._forward_train(feature,pa_mask,img_sizes,hbb,rbb,labels)
        else:
            return self._forward_test(feature,img_sizes)
        


