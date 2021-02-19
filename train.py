from tqdm import tqdm
from jittor import optim
import argparse
import sys
import glob
import pickle
import os
from tensorboardX import SummaryWriter
from dataset.dota import DOTA
from models.scrdet import SCRDet

CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']
EPOCHS=10

def train():
    jt.flags.use_cuda=1
    train_dataset = DOTA(data_dir="/mnt/disk/lxl/dataset/DOTA_CROP/train",
                        classnames=CLASSNAMES,
                        transforms=None,
                        num_workers=4,
                        shuffle=False,
                        batch_size=4)
    val_dataset = DOTA(data_dir="/mnt/disk/lxl/dataset/DOTA_CROP/val",
                        classnames=CLASSNAMES,
                        transforms=None,
                        num_workers=4,
                        shuffle=False,
                        batch_size=4)
    
    scrdet = SCRDet(n_class = len(CLASSNAMES)+1)

    optimizer = optim.SGD(scrdet.parameters(),momentum=0.9,lr=0.001)
    
    writer = SummaryWriter()
    
    for epoch in range(EPOCHS):
        scrdet.train()
        for batch_idx,(batch_imgs,img_sizes,hbb,rbb,ll,ids) in enumerate(train_dataset):
            pass
        

if __name__ == "__main__":
    train()