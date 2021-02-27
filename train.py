from tqdm import tqdm
import jittor as jt
from jittor import optim
from jittor.lr_scheduler import MultiStepLR
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
from utils.visualize import save_visualize_image

CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter']

CLASSNAMES = ['roundabout','tennis-court','swimming-pool', 'storage-tank',
                'soccer-ball-field','small-vehicle','ship',
                'plane', 'large-vehicle',
                'helicopter','harbor','ground-track-field',
                'bridge','basketball-court','baseball-diamond']

EPOCHS=20
NUM_GPUS = 1
BATCH_SIZE=1*NUM_GPUS
LR = 0.001 * BATCH_SIZE 
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 8

if jt.in_mpi:
    world_rank = jt.mpi.world_rank()
    is_main = (world_rank == 0)
else:
    is_main = True

save_checkpoint_path = "/mnt/disk/lxl/SCRDET_new"
data_dir = '/mnt/disk/lxl/dataset/DOTA_CROP'

def test(save_result=True,display_img=True):
    checkpoint_path = f'{save_checkpoint_path}/checkpoint_3.pkl'
    jt.flags.use_cuda=1
    val_dataset = DOTA(data_dir=f"{data_dir}/val",
                        classnames=CLASSNAMES,
                        transforms=val_transforms(),
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        batch_size=BATCH_SIZE)
    scrdet = SCRDet(classnames = CLASSNAMES)
    scrdet.load("test.pkl")
    # scrdet.load(checkpoint_path)
    scrdet.test_score_thresh=0.001
    scrdet.eval()
    results = []
    results_r = []
    for batch_idx,(batch_imgs,batch_masks,img_sizes,hbb,rbb,labels,ids) in tqdm(enumerate(val_dataset)):
        print(batch_imgs.mean())
        result,result_r = scrdet(batch_imgs,img_sizes,img_ids=ids)
        for i in range(len(ids)):
            pred_boxes,pred_scores,pred_labels = result[i]
            gt_boxes = hbb[i]
            pred_boxes_r,pred_scores_r,pred_labels_r = result_r[i]
            gt_boxes_r = rbb[i]
            gt_labels = labels[i]
            img_id = ids[i]
            results.append((img_id,pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy(),gt_boxes.numpy(),gt_labels.numpy()))
            results_r.append((img_id,pred_boxes_r.numpy(),pred_labels_r.numpy(),pred_scores_r.numpy(),gt_boxes_r.numpy(),gt_labels.numpy()))
            # print(pred_boxes)
            if display_img:
                save_visualize_image(data_dir,img_id,pred_boxes.numpy(),pred_scores.numpy(),pred_labels.numpy(),gt_boxes.numpy(),gt_labels.numpy(),CLASSNAMES)
        
    if save_result:
        pickle.dump((results,result_r),open(checkpoint_path.replace(".pkl","_results.pkl"),"wb"))
    mAP,_ = calculate_VOC_mAP(results,CLASSNAMES,use_07_metric=False)
    mAP_r,_ = calculate_VOC_mAP_r(results_r,CLASSNAMES,use_07_metric=False)
    print(mAP,mAP_r)

def train():
    jt.flags.use_cuda=1
    train_dataset = DOTA(data_dir=f"{data_dir}/trainval",
                        classnames=CLASSNAMES,
                        transforms=train_transforms(),
                        num_workers=NUM_WORKERS,
                        shuffle=True,
                        batch_size=BATCH_SIZE)
    val_dataset = DOTA(data_dir=f"{data_dir}/trainval",
                        classnames=CLASSNAMES,
                        transforms=val_transforms(),
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        batch_size=BATCH_SIZE)
    
    scrdet = SCRDet(classnames = CLASSNAMES)

    optimizer = optim.SGD(scrdet.parameters(),momentum=MOMENTUM,lr=LR,weight_decay=WEIGHT_DECAY)
    scrdet.load("test.pkl")
    scheduler = MultiStepLR(optimizer, milestones=[12*27000,16*27000], gamma=0.3)
    
    writer = SummaryWriter()
    
    for epoch in range(EPOCHS):
        scrdet.train()
        dataset_len  = len(train_dataset)

        for batch_idx,(batch_imgs,batch_masks,img_sizes,hbb,rbb,labels,ids) in tqdm(enumerate(train_dataset)):
            rpn_loc_loss, rpn_cls_loss, roi_loc_loss, \
            roi_cls_loss,roi_loc_loss_r,roi_cls_loss_r,\
            att_loss,total_loss = scrdet(batch_imgs,img_sizes,batch_masks,hbb,rbb,labels,img_ids=ids,batch_idx=batch_idx)
            
            optimizer.step(total_loss)
            scheduler.step()
            
            if is_main:
                writer.add_scalar('rpn_cls_loss', rpn_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('rpn_loc_loss', rpn_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('roi_loc_loss', roi_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('roi_cls_loss', roi_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('roi_loc_loss_r', roi_loc_loss_r.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('roi_cls_loss_r', roi_cls_loss_r.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('att_loss', att_loss.item(), global_step=dataset_len*epoch+batch_idx)
                writer.add_scalar('total_loss', total_loss.item(), global_step=dataset_len*epoch+batch_idx)
            
            # if batch_idx % 10 == 0:
            #     print("total_loss: %.4f"% total_loss.item())
        
        if is_main:
            os.makedirs(save_checkpoint_path,exist_ok=True)
            scrdet.save(f"{save_checkpoint_path}/checkpoint_{epoch}.pkl")

        scrdet.eval()
        results = []
        results_r = []
        for batch_idx,(batch_imgs,batch_masks,img_sizes,hbb,rbb,labels,ids) in tqdm(enumerate(val_dataset)):
            result,result_r = scrdet(batch_imgs,img_sizes,img_ids=ids,batch_idx=batch_idx)
            for i in range(len(ids)):
                pred_boxes,pred_scores,pred_labels = result[i]
                gt_boxes = hbb[i]
                pred_boxes_r,pred_scores_r,pred_labels_r = result_r[i]
                gt_boxes_r = rbb[i]
                gt_labels = labels[i]
                img_id = ids[i]
                results.append((img_id,pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy(),gt_boxes.numpy(),gt_labels.numpy()))
                results_r.append((img_id,pred_boxes_r.numpy(),pred_labels_r.numpy(),pred_scores_r.numpy(),gt_boxes_r.numpy(),gt_labels.numpy()))
            
        mAP,_ = calculate_VOC_mAP(results,CLASSNAMES,use_07_metric=True)
        mAP_r,_ = calculate_VOC_mAP_r(results_r,CLASSNAMES,use_07_metric=True)
        print(mAP,mAP_r)
        writer.add_scalar('map', mAP, global_step=epoch)
        writer.add_scalar('map_r', mAP_r, global_step=epoch)
        
def main():
    parser = argparse.ArgumentParser(description='Test a SCRDet network')
    parser.add_argument('--task',help='Task(train,test)',default='train',type=str)
    args = parser.parse_args()
    if args.task == 'test':
        test()
    elif args.task == 'train':
        train()
    else:
        print(f"No this task: {args.task}")

if __name__ == "__main__":
    main()