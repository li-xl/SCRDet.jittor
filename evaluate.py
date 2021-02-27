# import jittor as jt 
import numpy as np 
import os 
import glob 
import cv2 
import pickle
from PIL import Image 
from tqdm import tqdm
import jittor as jt
from utils.box_utils import convert_coordinate,convert_rotation_box
from utils.rotation_nms import rotate_nms
from dataset.dota import TESTDOTA
from dataset.transforms import val_transforms
from models.scrdet import SCRDet

CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']


nms_threshold_r = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.2, 'plane': 0.3,
                'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3}

nms_threshold_h = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
                'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
                'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
                'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3}

def get_image_list(img_dir):
    image_path = os.path.join(img_dir,"*.png")
    img_list = list(glob.glob(image_path))
    assert len(img_list)>0,"no imgs"
    return img_list

def build_images(img_list,width,height,stride_w,stride_h,save_path): 
    for i,img_path in enumerate(img_list):
        img_id = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path)
        imgH,imgW = img.shape[0],img.shape[1]
        
        imgH = max(imgH,height)
        imgW = max(imgW,width)

        temp = np.zeros((imgH,imgW,3),dtype=img.dtype)
        temp[:img.shape[0],:img.shape[1],:]=img 
        img = temp
        
        for hh in range(0,imgH,stride_h):
            if imgH-hh-1<height:
                hh_ = imgH-height
            else:
                hh_ = hh 
            for ww in range(0,imgW,stride_w):
                if imgW-ww-1<width:
                    ww_ = imgW-width
                else:
                    ww_ = ww 
                
                src_img = img[hh_:(hh_+height),ww_:(ww_+width),:]
                os.makedirs(save_path,exist_ok=True)
                save_file = os.path.join(save_path,f"{img_id}_{hh_}_{ww_}_{src_img.shape[0]}_{src_img.shape[1]}.png")
                cv2.imwrite(save_file,src_img)
        
        print(img_id,f'{i+1}/{len(img_list)}')

def run_eval(img_list,checkpoint_path,save_result_path):
    val_dataset = TESTDOTA(img_list=img_list,
                        transforms=val_transforms(),
                        num_workers=4,
                        shuffle=False,
                        batch_size=1)
    scrdet = SCRDet(classnames = CLASSNAMES)
    scrdet.load(checkpoint_path)
    scrdet.eval()
    
    results = []
    results_r = []

    for batch_idx,(batch_imgs,img_sizes,ids) in tqdm(enumerate(val_dataset)):
        result,result_r = scrdet(batch_imgs,img_sizes)
        for i in range(len(ids)):
            pred_boxes,pred_scores,pred_labels = result[i]
            pred_boxes_r,pred_scores_r,pred_labels_r = result_r[i]
            img_size = img_sizes[i]
            img_id = ids[i]
            img_size = jt.array(img_size).numpy()
            results.append((img_id,img_size,pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy()))
            results_r.append((img_id,pred_boxes_r.numpy(),pred_labels_r.numpy(),pred_scores_r.numpy()))
    
    os.makedirs(save_result_path,exist_ok=True)
    name = checkpoint_path.split("/")[-1].split(".")[0]
    save_file = os.path.join(save_result_path,f"{name}_result.pkl")
    pickle.dump((results,results_r),open(save_file,"wb"))

def nms(boxes,scores,thresh):
    boxes = jt.array(boxes)
    scores = jt.array(scores)
    dets = jt.contrib.concat([boxes,scores.unsqueeze(1)],dim=1)
    keep = jt.nms(dets,thresh)
    keep = np.where(keep.numpy())[0]
    return keep

def nms_rotate(boxes,scores,thresh):
    boxes = jt.array(convert_rotation_box(boxes,False))
    scores = jt.array(scores)
    dets = jt.contrib.concat([boxes,scores.unsqueeze(1)],dim=1)
    keep = rotate_nms(dets,thresh)
    keep = np.where(keep.numpy())[0]
    return keep

    
def merge(result_file,save_path):
    results_h,results_r = pickle.load(open(result_file,"rb"))


    results = {}
    for i, ((img_id,img_size,pred_boxes,pred_labels,pred_scores),(img_id1,pred_boxes_r,pred_labels_r,pred_scores_r)) in enumerate(zip(results_h,results_r)):
        assert img_id==img_id1,"no a same img"
        img_id,hh_,ww_,height,width = img_id.split("_")
        hh_,ww_,height,width = int(hh_),int(ww_),int(height),int(width)
        resize_w,resize_h = img_size
        
        if len(pred_boxes)>0:
            pred_boxes[:,0::2] *= (width/resize_w)
            pred_boxes[:,1::2] *= (height/resize_h)
            pred_boxes[:,0::2] += ww_
            pred_boxes[:,1::2] += hh_
        
        pred_boxes_r = convert_coordinate(pred_boxes_r,False)
        if len(pred_boxes_r)>0:
            pred_boxes_r[:,0::2] *= (width/resize_w)
            pred_boxes_r[:,1::2] *= (height/resize_h)
            pred_boxes_r[:,0::2] += ww_
            pred_boxes_r[:,1::2] += hh_

        if img_id not in results:
            results[img_id]={
                "box_r":[pred_boxes_r],
                "box":[pred_boxes],
                "score":[pred_scores],
                "score_r":[pred_scores_r],
                "label":[pred_labels],
                "label_r":[pred_labels_r]
            }
        else:
            results[img_id]["box_r"].append(pred_boxes_r)
            results[img_id]["box"].append(pred_boxes)
            results[img_id]["score_r"].append(pred_scores_r)
            results[img_id]["score"].append(pred_scores)
            results[img_id]["label"].append(pred_labels)
            results[img_id]["label_r"].append(pred_labels_r)
        # print(i,"/",len(results_r))
    
    save_results = {}
    save_results_r = {}
    for img_id,data in results.items():
        pred_boxes_r = np.concatenate(data["box_r"],axis=0)
        pred_scores_r = np.concatenate(data["score_r"],axis=0)
        pred_labels_r = np.concatenate(data["label_r"],axis=0)
        pred_boxes = np.concatenate(data["box"],axis=0)
        pred_scores = np.concatenate(data["score"],axis=0)
        pred_labels = np.concatenate(data["label"],axis=0)
        
        for c in np.unique(pred_labels):
            index = np.where(pred_labels==c)[0]
            boxes = pred_boxes[index,:]
            scores = pred_scores[index]
            classname = CLASSNAMES[c-1]
            inx = nms(boxes,scores,nms_threshold_h[classname])
            boxes = boxes[inx,:]
            scores = scores[inx]
            if classname not in save_results:
                save_results[classname]=[]
            for box,score in zip(boxes,scores):
                command = '%s %.3f %.1f %.1f %.1f %.1f\n'%(img_id,score,box[0],box[1],box[2],box[3])
                save_results[classname].append(command)

        for c in np.unique(pred_labels_r):
            index = np.where(pred_labels_r==c)[0]
            boxes = pred_boxes_r[index,:]
            scores = pred_scores_r[index]
            classname = CLASSNAMES[c-1]
            inx = nms_rotate(boxes,scores,nms_threshold_r[classname])
            boxes = boxes[inx,:]
            scores = scores[inx]
            if classname not in save_results_r:
                save_results_r[classname]=[]
            for box,score in zip(boxes,scores):
                command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%( img_id,
                                                                                 score,
                                                                                 box[0],box[1],box[2],box[3],
                                                                                 box[4],box[5],box[6],box[7])
                save_results_r[classname].append(command)

    for k,result in save_results.items():
        result = ''.join(result)
        save_file = os.path.join(save_path,"HBB")
        os.makedirs(save_file,exist_ok=True)
        save_file = os.path.join(save_file,f"Task2_{k}.txt")
        with open(save_file,"w") as f:
            f.write(result)
    
    for k,result in save_results_r.items():
        result = ''.join(result)
        save_file = os.path.join(save_path,"OBB")
        os.makedirs(save_file,exist_ok=True)
        save_file = os.path.join(save_file,f"Task1_{k}.txt")
        with open(save_file,"w") as f:
            f.write(result)

def eval():
    jt.flags.use_cuda=1
    epoch = 7
    img_dir = '/mnt/disk/lxl/dataset/DOTA/test/images'
    crop_images_path = '/mnt/disk/lxl/dataset/DOTA_CROP/test'
    save_result_path = '/mnt/disk/lxl/dataset/DOTA_CROP/test_result'
    checkpoint_path = f'/mnt/disk/lxl/SCRDET/checkpoint_{epoch}.pkl'
    result_file = f"{save_result_path}/checkpoint_{epoch}_result.pkl"

    # img_list = get_image_list(img_dir)
    # build_images(img_list=img_list,width=600,height=600,stride_w=450,stride_h=450,save_path=crop_images_path)
    
    # crop_img_list = get_image_list(crop_images_path)
    # run_eval(crop_img_list,checkpoint_path,save_result_path)
    merge(result_file,save_result_path)


if __name__ == '__main__':
    eval()
