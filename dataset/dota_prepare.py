import os
import numpy as np
import copy
import cv2
import sys
import json
import glob
sys.path.append("./")

from utils.box_utils import convert_rotation_box



CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']

def save_json(save_file,idx,objects,img_w,img_h,classnames):
    data = {
              "id":idx,
              "img_size":(img_h,img_w),
              "objects":[{"boxes":o[:8],"category":classnames[int(o[8])],"diffculty":int(o[9]) }for o in objects.tolist()]
           }
    json.dump(data,open(save_file,"w"))


def read_data(txt_file,classnames):
    all_data = []
    with open(txt_file) as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            if len(data)<10:continue 
            box = [float(d) for d in data[:8]]
            category = [classnames.index(data[8])]
            difficult = [int(data[9])]
            all_data.append(box+category+difficult)
    return all_data


def clip_image(file_idx, image, boxes_all, width, height, stride_w, stride_h,save_dir):
    print(file_idx)

    boxes_all = np.array(boxes_all)

    # fill useless boxes
    min_pixel = 5
    boxes_all_5 = convert_rotation_box(boxes_all[:, :8], False)

    boxes_all = boxes_all[np.logical_and(boxes_all_5[:, 2] > min_pixel, boxes_all_5[:, 3] > min_pixel), :]

    if boxes_all.shape[0] > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0 and (subImage.shape[0] > 5 and subImage.shape[1] > 5):

                    os.makedirs(os.path.join(save_dir, 'images'),exist_ok=True)
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                    cv2.imwrite(img, subImage)

                    os.makedirs(os.path.join(save_dir, 'labeltxt'),exist_ok=True)
                    save_file = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.json" % (file_idx, top_left_row, top_left_col))

                    save_json(save_file,"%s_%04d_%04d" % (file_idx, top_left_row, top_left_col), box[idx, :],subImage.shape[0], subImage.shape[1], CLASSNAMES)

def prepare_dota(part_name="val"):
    np.random.seed(0)
    data_dir = f"/mnt/disk/lxl/dataset/DOTA/{part_name}"
    save_dir = f"/mnt/disk/lxl/dataset/DOTA_CROP/{part_name}"
    img_h, img_w, stride_h, stride_w = 800, 800, 600, 600

    image_files = glob.glob(f"{data_dir}/images/*.png")
    for img_f in image_files:
        txt_f = img_f.replace("images","labelTxt-v1.0/labelTxt").replace("png","txt")
        img = cv2.imread(img_f,cv2.IMREAD_UNCHANGED)
        box = read_data(txt_f,CLASSNAMES)
        if len(box)>0:
            clip_image(img_f.split("/")[-1].strip('.png'), img, box, img_w, img_h, stride_w, stride_h,save_dir)

if __name__ == '__main__':
    prepare_dota(part_name="train")
