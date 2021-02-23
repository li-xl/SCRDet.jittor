from tqdm import tqdm
import jittor as jt
from jittor import optim
import argparse
import random
import sys
import glob
import pickle
import os
import numpy as np
from tensorboardX import SummaryWriter
from dataset.dota import DOTA
from dataset.transforms import train_transforms,val_transforms
from models.scrdet import SCRDet
from utils.ap_eval import calculate_VOC_mAP,calculate_VOC_mAP_r

CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']
    
threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
            'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.2, 'plane': 0.3,
            'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
            'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3,'container-crane':0.3}

EPOCHS=10
BATCH_SIZE=4
LR = 0.001 * BATCH_SIZE 
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 4

save_checkpoint_path = "/mnt/disk/lxl/SCRDET"

def train():
    jt.flags.use_cuda=1
    train_dataset = DOTA(data_dir="/mnt/disk/lxl/dataset/DOTA_CROP/train",
                        classnames=CLASSNAMES,
                        transforms=train_transforms(),
                        num_workers=NUM_WORKERS,
                        shuffle=True,
                        batch_size=BATCH_SIZE)
    val_dataset = DOTA(data_dir="/mnt/disk/lxl/dataset/DOTA_CROP/val",
                        classnames=CLASSNAMES,
                        transforms=val_transforms(),
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        batch_size=BATCH_SIZE)
    
    scrdet = SCRDet(classnames = CLASSNAMES,r_nms_thresh=threshold)

    optimizer = optim.SGD(scrdet.parameters(),momentum=MOMENTUM,lr=LR)
    
    writer = SummaryWriter()
    
    for epoch in range(EPOCHS):
        scrdet.train()
        dataset_len  = len(train_dataset)

        for batch_idx,(batch_imgs,batch_masks,img_sizes,hbb,rbb,labels,ids) in enumerate(train_dataset):
            rpn_loc_loss, rpn_cls_loss, roi_loc_loss, \
            roi_cls_loss,roi_loc_loss_r,roi_cls_loss_r,\
            att_loss,total_loss = scrdet(batch_imgs,img_sizes,batch_masks,hbb,rbb,labels)
            
            optimizer.step(total_loss)

            writer.add_scalar('rpn_cls_loss', rpn_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('rpn_loc_loss', rpn_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_loc_loss', roi_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_cls_loss', roi_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_loc_loss_r', roi_loc_loss_r.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_cls_loss_r', roi_cls_loss_r.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('att_loss', att_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('total_loss', total_loss.item(), global_step=dataset_len*epoch+batch_idx)
            
            if batch_idx % 10 == 0:
                print("total_loss: %.4f"% total_loss.item())

        os.makedirs(save_checkpoint_path,exist_ok=True)
        scrdet.save(f"{save_checkpoint_path}/checkpoint_{epoch}.pkl")

        scrdet.eval()
        results = []
        results_r = []
        for batch_idx,(batch_imgs,batch_masks,img_sizes,hbb,rbb,labels,ids) in tqdm(enumerate(val_dataset)):
            result,result_r = scrdet(batch_imgs,img_sizes)
            for i in range(len(ids)):
                pred_boxes,pred_scores,pred_labels = result[i]
                gt_boxes = hbb[i]
                pred_boxes_r,pred_scores_r,pred_labels_r = result_r[i]
                gt_boxes_r = rbb[i]
                gt_labels = labels[i]
                img_id = ids[i]
                results.append((img_id,pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy(),gt_boxes.numpy(),gt_labels.numpy()))
                results_r.append((img_id,pred_boxes_r.numpy(),pred_labels_r.numpy(),pred_scores_r.numpy(),gt_boxes_r.numpy(),gt_labels.numpy()))
            
        mAP,_ = calculate_VOC_mAP(results,CLASSNAMES,use_07_metric=False)
        mAP_r,_ = calculate_VOC_mAP(results_r,CLASSNAMES,use_07_metric=False)
        writer.add_scalar('map', mAP, global_step=epoch)
        writer.add_scalar('map_r', mAP_r, global_step=epoch)
        

if __name__ == "__main__":
    train()