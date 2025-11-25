import os
import json
import cv2
import torch
import numpy as np
from collections import defaultdict
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from collections import Counter

image_dir = "/cpfs/user/wenzhuangwang/FICGen/datasets/dior/train/"
label_dir = "/cpfs/user/wenzhuangwang/FICGen/datasets/dior/labels/"

anno_dir = os.path.join('/cpfs/user/wenzhuangwang/FICGen/datasets/dior/annotations/')
listin = []
annotations_ = json.load(open(os.path.join(anno_dir, "instances_train.json"),"r"))
images_items = annotations_["images"]
annos_items = annotations_["annotations"]
cates_items = annotations_["categories"]
catemap = {}
for cate in cates_items:
    catemap[cate["id"]] = cate["name"]

train = {}
for item in images_items:
    train[item["id"]] = item["file_name"]

annototals = []
annos_ = defaultdict(list)

def get_area_thresholds(annos):
    areas = []
    for anno in annos:
        xmin, ymin, w, h, _ = anno
        xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
        
        xmax = xmin + w
        ymax = ymin + h
     
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        
        bbox_area = bbox_width * bbox_height
        areas.append(bbox_area)
    small_thresh = np.percentile(areas, 33)
    large_thresh = np.percentile(areas, 66)
    return small_thresh, large_thresh, areas

for item in annos_items:
    image_id = item["image_id"]
    filename = train[image_id].replace(".tif", ".jpg")
    annos_[filename].append(item["bbox"] + [item["category_id"]])
    annototals.append(item["bbox"] + [item["category_id"]])
small_thresh, large_thresh, _ = get_area_thresholds(annototals)
output_dir = "voc/"
        
    
def anno_parser(anno_path, dataset_name="DIOR"):
    if dataset_name == 'DIOR':
        root = ET.parse(anno_path).getroot()
        name_list = []
        bbox_list = []
        for node in root.findall('object'):
            name = node.find('name').text
            name_list.append(name)
            bbox_node = node.find('bndbox')
            bbox = [int(child.text) for child in bbox_node]
            bbox_list.append([name] + bbox)    
        return bbox_list
    else:
        return None, None

os.makedirs(output_dir, exist_ok=True)
print(len(os.listdir(image_dir)))
counter = Counter()
for image_name in os.listdir(image_dir):

    if not image_name.endswith(".jpg"):
        continue
        
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    anno_path = annos_[image_name]
    
    classes = []
    bndboxes = []
    obndboxes= []
    annos = anno_path
    for i,anno in enumerate(annos):
        scale = None
        xmin, ymin, w, h, cate = anno
        xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
        
        xmax = xmin + w
        ymax = ymin + h
        class_name = catemap[cate]

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        
        bbox_area = bbox_width * bbox_height
      
   #     if bbox_area <= small_thresh:
   #         scale = "small"
   #     elif bbox_area <= large_thresh:
   #         scale = "medium"
   #     else:
   #         scale = "large"
            
        image_area = image_width * image_height
        bbox_ratio = bbox_area / image_area
        
        if bbox_ratio < 0.0005:
            continue
   #     assert scale is not None, "shit scale!!"
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        counter[class_dir] += 1
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        output_image_name = f"{image_name[:-4]}_{i}.jpg"
        output_image_path = os.path.join(class_dir, output_image_name)
        
        cv2.imwrite(output_image_path, cropped_image)
        
        
print(counter)        
print("***所有目标已成功抠出并保存到对应类别的文件夹中.***")
