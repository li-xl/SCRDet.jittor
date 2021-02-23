import jittor as jt 
from jittor import nn 

from models.resnet import resnet50
from models.fpn import SCRDetFPN
from models.rpn import RegionProposalNetwork,AnchorTargetCreator,ProposalCreator,ProposalTargetCreator
from models.roi_head import RoIHead
from models.loss import smooth_l1_loss_rcnn,iou_smooth_l1_loss_rcnn_r,attention_loss
from utils.box_utils import loc2bbox_r,loc2bbox
from utils.rotation_nms import rotate_nms


class SCRDet(nn.Module):

    def __init__(self,classnames,r_nms_thresh):
        super(SCRDet,self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.fpn = SCRDetFPN(anchor_stride=8,out_dim=512,ratio=16)
        self.n_class = len(classnames)+1
        self.classnames = classnames

        self.rpn = RegionProposalNetwork(in_channels=512, 
                                        mid_channels=512, 
                                        ratios=[0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0],
                                        anchor_scales= [0.0625, 0.125, 0.25, 0.5, 1., 2.0], 
                                        feat_stride=8,
                                        nms_thresh=0.7,
                                        n_train_pre_nms=12000,
                                        n_train_post_nms=2000,
                                        n_test_pre_nms=6000,
                                        n_test_post_nms=1000,
                                        min_size=16,)

        self.anchor_target_creator = AnchorTargetCreator(n_sample=512,
                                                         pos_iou_thresh=0.7, 
                                                         neg_iou_thresh=0.3,
                                                         pos_ratio=0.5)

        self.proposal_target_creator = ProposalTargetCreator(n_sample=512,
                                                             pos_ratio=0.25, 
                                                             pos_iou_thresh=0.5,
                                                             neg_iou_thresh_hi=0.5, 
                                                             neg_iou_thresh_lo=0.0)
        
        self.head = RoIHead(in_channels=512,
                            n_class=self.n_class,
                            roi_size=14,
                            spatial_scale=1.0,
                            sampling_ratio=0)

        self.rpn_sigma = 3.0
        self.roi_sigma = 1.0
        self.test_score_thresh = 0.001
        self.test_nms_thresh = 0.3
        self.test_nms_thresh_r = r_nms_thresh

    def _forward_train(self,feature,pa_mask,img_sizes,batch_masks,hbb,rbb,labels):
        N = feature.shape[0]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(feature, img_sizes)
        
        sample_rois = []
        gt_roi_locs = []
        gt_roi_locs_r = []
        gt_roi_labels = []
        sample_roi_indexs = []
        gt_rpn_locs = []
        gt_rpn_labels = []
        for i in range(N):
            index = jt.where(roi_indices == i)[0]
            roi = rois[index,:]
            box = hbb[i]
            box_r = rbb[i]
            label = labels[i]
            sample_roi, gt_roi_loc, gt_roi_loc_r,gt_roi_label = self.proposal_target_creator(roi,box,box_r,label)
            sample_roi_index = i*jt.ones((sample_roi.shape[0],))
            
            sample_rois.append(sample_roi)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_locs_r.append(gt_roi_loc_r)
            sample_roi_indexs.append(sample_roi_index)
            
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(box,anchor,img_sizes[i])
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
        rpn_loc_loss = smooth_l1_loss_rcnn(rpn_locs,gt_rpn_locs,gt_rpn_labels,self.rpn_sigma)
        rpn_cls_loss = nn.cross_entropy_loss(rpn_scores[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
        gt_roi_locs_r = jt.contrib.concat(gt_roi_locs_r,dim=0)
        gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)

        n_sample = bbox_pred_h.shape[0]
        roi_cls_loc = bbox_pred_h.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[jt.arange(0, n_sample).int32(), gt_roi_labels]
        roi_loc_loss = smooth_l1_loss_rcnn(roi_loc,gt_roi_locs,gt_roi_labels,self.roi_sigma)
        roi_cls_loss = nn.cross_entropy_loss(cls_score_h, gt_roi_labels)

        n_sample_r = bbox_pred_r.shape[0]
        roi_cls_loc_r = bbox_pred_r.view(n_sample, -1, 5)
        roi_loc_r = roi_cls_loc_r[jt.arange(0, n_sample_r).int32(), gt_roi_labels]
        roi_loc_loss_r = iou_smooth_l1_loss_rcnn_r(roi_loc_r,gt_roi_locs_r,gt_roi_labels,sample_rois,self.roi_sigma)
        roi_cls_loss_r = nn.cross_entropy_loss(cls_score_r, gt_roi_labels)

        # ------------------ Attention losses -------------------#
        att_loss = attention_loss(batch_masks,pa_mask)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss,roi_loc_loss_r,roi_cls_loss_r,att_loss]
        losses = losses + [sum(losses)]
        return losses
    
    def _forward_test(self,features,img_size,batch_size):
        rpn_locs, rpn_scores, rois, roi_indices, _ = self.rpn(features, img_size)
        bbox_pred_h,cls_score_h,bbox_pred_r,cls_score_r = self.head(features, rois, roi_indices)
        return self._build_result(batch_size,img_size,rois,roi_indices,bbox_pred_h,bbox_pred_r,cls_score_h,cls_score_r)
    
    def _build_result(self,batch_size,image_sizes,rois,roi_indices,bbox_pred_h,bbox_pred_r,cls_score_h,cls_score_r):
        bbox_pred_h = bbox_pred_h.reshape([bbox_pred_h.shape[0],-1,4])
        bbox_pred_r = bbox_pred_r.reshape([bbox_pred_r.shape[0],-1,5])
        rois = rois.unsqueeze(1).repeat(1,bbox_pred_h.shape[1],1)
        cls_bboxes = loc2bbox(rois.reshape(-1,4),bbox_pred_h.reshape(-1,4))
        cls_bboxes_r = loc2bbox_r(rois.reshape(-1,4),bbox_pred_r.reshape(-1,5))

        cls_bboxes = cls_bboxes.reshape(bbox_pred_h.shape)
        cls_bboxes_r = cls_bboxes_r.reshape(bbox_pred_r.shape)

        probs = nn.softmax(cls_score_h,dim=-1)
        probs_r = nn.softmax(cls_score_r,dim=-1)

        n_class = bbox_pred_h.shape[1]
        results = []
        results_r = []
        for i in range(batch_size):
            index = jt.where(roi_indices==i)[0]
            score = probs[index,:]
            score_r = probs_r[index,:]
            bbox = cls_bboxes[index,:,:]
            bbox_r = cls_bboxes_r[index,:,:]
            boxes = []
            scores = []
            labels = []
            boxes_r = []
            scores_r = []
            labels_r = []
            for j in range(1,n_class):
                classname = self.classnames[j-1]
                bbox_j = bbox[:,j,:]
                score_j = score[:,j]
                mask = jt.where(score_j>self.test_score_thresh)[0]
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]
                dets = jt.contrib.concat([bbox_j,score_j.unsqueeze(1)],dim=1)
                keep = jt.nms(dets,self.test_nms_thresh)
                bbox_j = bbox_j[keep]
                score_j = score_j[keep]
                label_j = jt.ones_like(score_j).int32()*j
                boxes.append(bbox_j)
                scores.append(score_j)
                labels.append(label_j)

                bbox_j_r = bbox_r[:,j,:]
                score_j_r = score_r[:,j]
                mask_r = jt.where(score_j_r>self.test_score_thresh)[0]
                bbox_j_r = bbox_j_r[mask_r,:]
                score_j_r = score_j_r[mask_r]
                dets_r = jt.contrib.concat([bbox_j_r,score_j_r.unsqueeze(1)],dim=1)
                keep_r = rotate_nms(dets_r,self.test_nms_thresh_r[classname])
                bbox_j_r = bbox_j_r[keep_r]
                score_j_r = score_j_r[keep_r]
                label_j_r = jt.ones_like(score_j_r).int32()*j
                boxes_r.append(bbox_j_r)
                scores_r.append(score_j_r)
                labels_r.append(label_j_r)
            
            boxes = jt.contrib.concat(boxes,dim=0)
            scores = jt.contrib.concat(scores,dim=0)
            labels = jt.contrib.concat(labels,dim=0)
            boxes_r = jt.contrib.concat(boxes_r,dim=0)
            scores_r = jt.contrib.concat(scores_r,dim=0)
            labels_r = jt.contrib.concat(labels_r,dim=0)
            results.append((boxes,scores,labels))
            results_r.append((boxes_r,scores_r,labels_r))

        return results,results_r


    def execute(self,batch_imgs,img_sizes,batch_masks=None,hbb=None,rbb=None,labels=None):
        # backbone
        N = batch_imgs.shape[0]
        C3,C4 = self.backbone(batch_imgs)
        
        feature,pa_mask = self.fpn(C3,C4)

        if self.is_training():
            assert batch_masks is not None and hbb is not None and rbb is not None and labels is not None, "Model must has ground truth"
            return self._forward_train(feature,pa_mask,img_sizes,batch_masks,hbb,rbb,labels)
        else:
            return self._forward_test(feature,img_sizes,N)
        


