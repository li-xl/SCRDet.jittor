import jittor as jt 
from jittor import nn 
from utils.roi_align import ROIAlign

class RoIHead(nn.Module):
    def __init__(self,in_channels,n_class, roi_size, spatial_scale,sampling_ratio):
        super(RoIHead,self).__init__()

        self.roi = ROIAlign((roi_size, roi_size),spatial_scale,sampling_ratio=sampling_ratio)

        self.fc1 = nn.Linear(roi_size*roi_size*512,1024)
        self.fc2 = nn.Linear(1024,1024)

        self.cls_score_h = nn.Linear(1024,n_class)
        self.bbox_pred_h = nn.Linear(1024,n_class*4)

        self.cls_score_r = nn.Linear(1024,n_class)
        self.bbox_pred_r = nn.Linear(1024,n_class*5)


    def execute(self,x,rois,roi_indices):
        # fc layers
        indices_and_rois = jt.contrib.concat([roi_indices.unsqueeze(1), rois], dim=1)
        x = self.roi(x, indices_and_rois)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)

        # horizen branch
        cls_score_h = self.cls_score_h(x)
        bbox_pred_h = self.bbox_pred_h(x)

        # rotation branch
        cls_score_r = self.cls_score_r(x)
        bbox_pred_r = self.bbox_pred_r(x)
        
        return bbox_pred_h,cls_score_h,bbox_pred_r,cls_score_r
