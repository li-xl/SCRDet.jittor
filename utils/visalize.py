import cv2
import numpy as np 


def draw_box(img,coordinates):
    point = np.array(coordinates,dtype=np.float32).reshape(-1,2).astype(int)
    color=(255, 0, 0)
    thickness=1
    img = cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)
    return img

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