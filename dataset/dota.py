from jittor import dataset
import glob
import json
import os
from PIL import Image
import numpy as np

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
        ori_img_size = img.size

        anno = json.load(open(anno_path))["objects"]
        boxes = np.array([a["boxes"] for a in anno],dtype=np.float32)
        labels = np.array([self.classnames.index(a["category"])+1 for a in anno],dtype=np.int32)
        
        if self.transforms is not None:
            img,boxes = self.transforms(img,boxes)
        return img,boxes,labels,ori_img_size,img_idx

    def collate_batch(self,batch):
        pass

