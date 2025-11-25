import os
import json
from PIL import Image
import numpy as np
import cv2

import random
test = json.load(open("ruod/annotations/instances_test.json","r"))

images = test["images"]
annos = test["annotations"]
cates = test["categories"]
from collections import defaultdict

image2anno = defaultdict(list)
anno2image = {}

for i in images:
    anno2image[i["id"]]=i["file_name"]
    
for anno in annos:
    image2anno[anno2image[anno["image_id"]]].append([anno["category_id"]]+anno["bbox"])
    
catemap = {}
for i in cates:
    catemap[i["id"]] = i["name"]
    
#palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
#               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
#               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
#               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
#               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
#               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
#               (255, 179, 240), (0, 125, 92)]
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]


#categories = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

categories = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']
#categories = ["vehicle", "baseballfield", "groundtrackfield", "windmill", "bridge", \
#                    "overpass", "ship", "airplane", "tenniscourt", "airport", \
#                    "expressway-service-area", "basketballcourt", "stadium", "storagetank", "chimney", \
#                    "dam", "expressway-toll-station", "golffield", "trainstation", "harbor"]
#categories = ["motorcycle", "person", "bus", "truck", "car"]
#categories = ["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup","dog","motorbike", "people", "table"]
#categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
#                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

#categories = ["person","rider", "car", "truck", 
#              "bus", "train", "motorcycle","bicycle"]
names = ["007910.jpg"]
names = [name for name in names if name.endswith("jpg")]
for name in names:
 
    image_ = Image.open(os.path.join("/cpfs/user/wenzhuangwang/CC-Diff-main/generated_images/ruod_sd15_new_up16/",name)).convert("RGB")
 #   width, height = image_.size
    width, height = Image.open(os.path.join("/cpfs/user/wenzhuangwang/CC-Diff-main/datasets/ruod/test",name)).convert("RGB").size
    image_ = image_.resize([512,512])

    labels = image2anno[name]

  #  image_tmp = np.array(image_, dtype=np.uint8)
    image_tmp = np.ones((512, 512,3), dtype=np.uint8) * 255
    
    boxes=[]
    for label in labels:

        tmp = label[1:]
        x1 = int(tmp[0])
        y1 = int(tmp[1])
        w = int(tmp[2])
        h = int(tmp[3])
        x2 = int(x1+w)
        y2 = int(y1+h)
        
        x1 = int(x1/width*512) 
        y1 = int(y1/height*512)
        x2 = int(x2/width*512)
        y2 = int(y2/height*512)
        
        xn1 = x1 
        yn1 = y1
        xn2 = x2
        yn2 = y2
        
        boxes.append([xn1,yn1,xn2,yn2])
    classes = [catemap[i[0]] for i in labels]
    #print(boxes, classes)

    for box,label in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = palette[categories.index(label)]
        cv2.rectangle(image_tmp, (x1,y1),(x2,y2), color,2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,1.0,2)
        cv2.rectangle(image_tmp, (x1,y1-text_height),(x1+text_width,y1),color,-1)
        cv2.putText(image_tmp,label,(x1,y1 - text_height+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)
    image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_RGB2BGR)

    cv2.imwrite("./{}".format(name),image_tmp)
    





