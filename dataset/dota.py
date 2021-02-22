from jittor import dataset
import glob
import json
import os
import cv2
from PIL import Image
import numpy as np
from utils.box_utils import convert_rotation_box,convert_horizen_box,convert_coordinate

def get_mask(h,w, boxes):
    boxes = convert_coordinate(boxes,False)
    mask = np.zeros([h, w])
    for b in boxes:
        b = np.reshape(b[0:-1], [4, 2])
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=0)
    return np.array(mask, np.float32)

class DOTA(dataset.Dataset):
    def __init__(self,data_dir,classnames,transforms=None,num_workers=1,shuffle=False,batch_size=1):
        super(DOTA,self).__init__(num_workers=num_workers,shuffle=shuffle,batch_size=batch_size)
        self.data_dir = data_dir
        self.classnames = classnames
        self.transforms = transforms

        images_dir = os.path.join(data_dir,"images")
        label_dir = os.path.join(data_dir,"labeltxt")

        images = [i for i in os.listdir(images_dir) if 'png' in i]
        labels = [i for i in os.listdir(label_dir) if 'json' in i]
        
        assert len(images)==len(labels)
        
        self.image_ids = [i.strip(".png") for i in images]
        self.total_len = len(images)

    def __getitem__(self,index):
        img_idx = self.image_ids[index]
        img_path = os.path.join(self.data_dir,"images",f"{img_idx}.png")
        anno_path = os.path.join(self.data_dir,"labeltxt",f"{img_idx}.json")

        img = Image.open(img_path)
        anno = json.load(open(anno_path))["objects"]
        
        boxes = np.array([a["boxes"] for a in anno],dtype=np.float32)
        labels = np.array([self.classnames.index(a["category"])+1 for a in anno],dtype=np.int32)
        
        if self.transforms is not None:
            img,boxes = self.transforms(img,boxes)
        
        h,w = img.shape[-2],img.shape[-1]
        
        horizen_boxes = convert_horizen_box(boxes)
        rotation_boxes = convert_rotation_box(boxes)
        mask = get_mask(h,w,rotation_boxes)

        return img,horizen_boxes,rotation_boxes,labels,(h,w),mask,img_idx

    def collate_batch(self,batch):
        imgs = []
        masks = []
        img_sizes = []
        hbb = []
        rbb = []
        ll = []
        ids = []
        max_size = np.array([0,0])
        for img,horizen_boxes,rotation_boxes,labels,img_size,mask,img_idx in batch:
            imgs.append(img)
            masks.append(mask)
            img_sizes.append(img_size)
            hbb.append(horizen_boxes)
            rbb.append(rotation_boxes)
            ll.append(labels)
            ids.append(img_idx)
            max_size = np.maximum(max_size,img_size)
        batch_imgs = np.zeros((len(imgs),3,max_size[1],max_size[0]),dtype=np.float32)
        batch_masks = np.zeros((len(imgs),1,max_size[1],max_size[0]),dtype=np.float32)
        for i,(img,mask,size) in enumerate(zip(imgs,masks,img_sizes)):
            batch_imgs[i,:,:size[1],:size[0]]=img
            batch_masks[i,:,:size[1],:size[0]]=mask

        return batch_imgs,batch_masks,img_sizes,hbb,rbb,ll,ids


