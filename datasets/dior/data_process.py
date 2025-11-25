import os
import json
import re
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from overlap import create_masks_with_overlap
thr = 15
test = os.listdir("test")
test = [i for i in test if i.endswith("jpg")]
caproot = "metadata_test_origin.jsonl"
captions = open(caproot,"r").readlines()
anno_path = "annotations/HBB/"
caproot = "/cpfs/user/wenzhuangwang/PixArt-alpha/output/dior/"
listin = []
for item in captions:
    dictin = {}
    dictin = eval(item)
    filename = dictin["file_name"]
    if filename not in test:
        continue
    caption = open(os.path.join(caproot, filename.split(".")[0]+".txt"),"r").readlines()
    caption = [cap.strip() for cap in caption if cap.strip()][0]
    root = ET.parse(os.path.join(anno_path, filename.split(".")[0]+".xml")).getroot()
    
    rootb = ET.parse(os.path.join("annotations/HBB/", filename.split(".")[0]+".xml")).getroot()
    width,height = Image.open(os.path.join("test", filename)).size
    classes = []
    bndboxes = []
    obndboxes= []
   
    for node in root.findall('object'):
        name = node.find('name').text
        bndbox_node = node.find('bndbox')
        bndbox = [int(child.text) for child in bndbox_node]
        cate = name
        xmin, ymin, xmax, ymax = bndbox
        xmin = xmin / width
        ymin = ymin / height
        xmax = xmax / width
        ymax = ymax / height
        cate = cate.lower()
       
        classes.append(cate)
        bndboxes.append([xmin, ymin, xmax, ymax])
        obndboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
    if len(classes) > thr:
        classes = classes[:thr]
        bndboxes = bndboxes[:thr]
        obndboxes = obndboxes[:thr]
    while len(classes) < thr:
        classes.append("")
        bndboxes.append([0,0,0,0])
        obndboxes.append([0,0,0,0,0,0,0,0])
    
with open("test/metadata.jsonl","w", encoding="utf-8") as f:
    for item in listin:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        
        
    