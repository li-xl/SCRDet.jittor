import cv2
import numpy as np 
import os

def calculate_ious(gt_boxes,box):

    in_w = np.minimum(gt_boxes[:,2],box[2]) - np.maximum(gt_boxes[:,0],box[0])
    in_h = np.minimum(gt_boxes[:,3],box[3]) - np.maximum(gt_boxes[:,1],box[1])

    in_w = np.maximum(in_w,0)
    in_h = np.maximum(in_h,0)
    
    inter = in_w*in_h 

    area1 = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    area2 = (box[2]-box[0])*(box[3]-box[1])
    union = area1+area2-inter
    ious = inter / union
    jmax = np.argmax(ious)
    maxiou = ious[jmax]
    return maxiou,jmax
    
def draw_box_r(img,coordinates,text,color):
    point = np.array(coordinates,dtype=np.float32).reshape(-1,2).astype(int)
    thickness=1
    img = cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 


def draw_boxes(img,boxes,labels,classnames,scores=None, color=(0,0,0)):
    if scores is None:
        scores = ['']*len(labels) 
    for box,score,label in zip(boxes,scores,labels):
        box = [int(i) for i in box]
        text = classnames[label-1]+(f': {score:.2f}' if not isinstance(score,str) else score)
        img = draw_box(img,box,text,color)
    return img

def visualize_result(img_file,
                     pred_boxes,
                     pred_scores,
                     pred_labels,
                     gt_boxes,
                     gt_labels,
                     classnames,
                     iou_thresh=0.5,
                     miss_color=(255,0,0),
                     wrong_color=(0,255,0),
                     surplus_color=(0,0,255),
                     right_color=(0,255,255)):
    
    img = cv2.imread(img_file)

    detected = [False for _ in range(len(gt_boxes))]
    miss_boxes = []
    wrong_boxes = []
    surplus_boxes = []
    right_boxes = []

    # sort the box by scores
    ind = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[ind,:]
    pred_scores = pred_scores[ind]
    pred_labels = pred_labels[ind]

    # add background
    classnames = ['background']+classnames

    for box,score,label in zip(pred_boxes,pred_scores,pred_labels):
        ioumax = 0.
        if len(gt_boxes)>0:
            ioumax,jmax = calculate_ious(gt_boxes,box)
        if ioumax>iou_thresh:
            if not detected[jmax]:
                detected[jmax]=True
                if label == gt_labels[jmax]:
                    right_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
                else:
                    wrong_boxes.append((box,f'{classnames[label]}->{classnames[gt_labels[jmax]]}'))
            else:
                surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
        else:
            surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
    
    for box,label,d in zip(gt_boxes,gt_labels,detected):
        if not d:
            miss_boxes.append((box,f'{classnames[label]}'))
    
    colors = [miss_color]*len(miss_boxes) + [wrong_color]*len(wrong_boxes) + [right_color]*len(right_boxes) + [surplus_color]*len(surplus_boxes)

    boxes = miss_boxes + wrong_boxes + right_boxes + surplus_boxes
    
    for (box,text),color in zip(boxes,colors):
        img = draw_box(img,box,text,color)
    
    # draw colors
    colors = [right_color,wrong_color,miss_color,surplus_color]
    texts = ['Detect Right','Detect Wrong Class','Missed Ground Truth','Surplus Detection']
    for i,(color,text) in enumerate(zip(colors,texts)):
        img = cv2.rectangle(img=img, pt1=(0,i*30), pt2=(60,(i+1)*30), color=color, thickness=-1)
        img = cv2.putText(img=img, text=text, org=(70,(i+1)*30-5), fontFace=0, fontScale=0.8, color=color, thickness=2)
    return img


def find_dir(data_dir,img_id):
    return f"{data_dir}/val/images/{img_id}.png"

def save_visualize_image(data_dir,img_id,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames):
    img_file =  find_dir(data_dir,img_id)

    img = visualize_result(img_file,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames)

    os.makedirs('test_imgs',exist_ok=True)
    cv2.imwrite(f'test_imgs/{img_id}.jpg',img)

def visualize_rpn(img_id,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels):
    if isinstance(img_id,(tuple,list)):
        img_id = img_id[0]
    data_dir = "/mnt/disk/lxl/dataset/DOTA_CROP"
    img_file = f"{data_dir}/train/images/{img_id}.png"
    if not os.path.exists(img_file):
        img_file = f"{data_dir}/train/images/{img_id}.png"

    CLASSNAMES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship','tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field','roundabout', 'harbor',
              'swimming-pool', 'helicopter', 'container-crane']
    img = visualize_result(img_file,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames)
    


def visualize_attention(img_id,pa_mask,gt_mask):
    if isinstance(img_id,(tuple,list)):
        img_id = img_id[0]
    data_dir = "/mnt/disk/lxl/dataset/DOTA_CROP"
    img_file = f"{data_dir}/train/images/{img_id}.png"
    if not os.path.exists(img_file):
        img_file = f"{data_dir}/val/images/{img_id}.png"
    img = cv2.imread(img_file).astype(np.float32)
    os.makedirs("tmp",exist_ok=True)
    if hasattr(pa_mask,"numpy"):
        pa_mask = pa_mask.numpy()
    if gt_mask is not None and hasattr(gt_mask,"numpy"):
        gt_mask = gt_mask.numpy()
        gt_mask = gt_mask[0,0]*255

    pa_mask = pa_mask[0,1]
    pa_mask = 255*pa_mask
    
    
    h,w = pa_mask.shape
    img = cv2.resize(img,(w,h))
    if gt_mask is not None:
        result = np.zeros((h,w*3,3))
        result[:,:w,:]=img
        result[:,w:2*w,:]=pa_mask[:,:,None]
        gt_mask = cv2.resize(gt_mask,(w,h))
        result[:,2*w:,:]=gt_mask[:,:,None]
    else:
        result = np.zeros((h,w*2,3))
        result[:,:w,:]=img
        result[:,w:,:]=pa_mask[:,:,None]
    img = result
    img = img.astype(np.uint8)
    cv2.imwrite(f"tmp/{img_id}.png",img)  
    
    
def read_txt(txt_file):
    with open(txt_file) as f:
        data = [line.strip().split(" ") for line in f.readlines() if len(line.strip().split(" "))==10]
        boxes = [d[:8] for d in data]
        categories = [d[8] for d in data]
        others = [d[9] for d in data]
        return boxes,categories,others
    
def test():
    img_file = "/mnt/disk/lxl/dataset/DOTA/train/images/P0000.png"
    txt_file = "/mnt/disk/lxl/dataset/DOTA/train/labelTxt-v1.0/labelTxt/P0000.txt"
    img = cv2.imread(img_file)
    boxes,categories,others = read_txt(txt_file)
    for box in boxes:
        img = draw_box(img,box)
    # cv2.fillConvexPoly(img,np.array([10,10,20,20,30,505,23,23]).reshape(-1,2),color=(0,0,0))
    cv2.imwrite("test.jpg",img)

if __name__ == "__main__":
    test()