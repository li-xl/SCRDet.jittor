#coding=utf-8
import jittor as jt
import numpy as np 
import random
import jittor.transform as T 
from PIL import Image
from utils.box_utils import bbox8_flip,bbox8_resize

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image,target = t(image, target)
        return image, target
    
class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow,oh)

    def __call__(self, image, target=None):
        ori_size = image.size
        size = self.get_size(image.size)
        image = image.resize(size,Image.BILINEAR)
        if target is not None:
            target = bbox8_resize(target,ori_size,image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if target is not None:
                target = bbox8_flip(target,image.size)
        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            image = np.array(image).transpose((2,0,1))/255.0
        return image, target

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = np.array(mean).reshape(3,1,1)
        self.std = np.array(std).reshape(3,1,1)
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = (image-self.mean)/self.std
        return image, target
    
    
def build_transforms(min_size=800,
                     max_size=800,
                     flip_horizontal_prob=0.5,
                     mean=[102.9801, 115.9465, 122.7717],
                     std = [1.,1.,1.],
                     to_bgr255=True):
    

    transform = Compose([
            Resize(min_size, max_size),
            RandomHorizontalFlip(flip_horizontal_prob),
            ToTensor(),
            Normalize(mean=mean, std=std, to_bgr255=to_bgr255),
        ])
    return transform

def train_transforms():
    return build_transforms()#flip_horizontal_prob=0.5,mean=[123.68, 116.779, 103.939],std=[0.229, 0.224, 0.225])

def val_transforms():
    return build_transforms(flip_horizontal_prob=0.0)#,mean=[123.68, 116.779, 103.939],std=[0.229, 0.224, 0.225])