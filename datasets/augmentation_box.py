import os, json
import cv2
import numpy as np
from PIL import Image
import random
import json
from collections import defaultdict
def calculate_area(box):
    """计算边框的面积"""
    return (box[2] - box[0]) * (box[3] - box[1])

def calculate_aspect_ratio(box):
    """计算边框的宽高比（宽度/高度）"""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width / height if height != 0 else float('inf')  # 避免除零错误

def find_closest_box(given_box, candidate_boxes, area_weight=1.0, aspect_ratio_weight=1.0):
    """
    从候选边框列表中找出与给定边框面积和宽高比最接近的边框。

    参数:
        given_box (list): 给定的边框 [x1, y1, x2, y2]。
        candidate_boxes (list): 候选边框列表，每个边框为 [x1, y1, x2, y2]。
        area_weight (float): 面积差的权重。
        aspect_ratio_weight (float): 宽高比差的权重。

    返回:
        closest_box (list): 最接近的边框 [x1, y1, x2, y2]。
        min_score (float): 最小的综合评分。
    """
    # 计算给定边框的面积和宽高比
    given_area = calculate_area(given_box)
    given_aspect_ratio = calculate_aspect_ratio(given_box)

    # 初始化最小评分和最接近的边框
    min_score = float('inf')
    closest_box = None

    # 遍历候选边框列表
    for box in candidate_boxes:
        # 计算候选边框的面积和宽高比
        candidate_area = calculate_area(box)
        candidate_aspect_ratio = calculate_aspect_ratio(box)

        # 计算面积差和宽高比差
        area_diff = abs(candidate_area - given_area)
        aspect_ratio_diff = abs(candidate_aspect_ratio - given_aspect_ratio)

        # 计算综合评分
        score = area_weight * area_diff + aspect_ratio_weight * aspect_ratio_diff

        # 更新最接近的边框
        if score < min_score:
            min_score = score
            closest_box = box

    return closest_box, min_score

def is_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max<=y1_min:
        return False
    return True


def random_flip(boxes, classes, probability=0.8, width=512):
    if random.random() < probability:
        for box in boxes:
            x1, y1, x2, y2 = box
            box[0], box[2] = width - x2, width - x1
    return boxes, classes

def safe_random_shift(boxes, classes, max_shift=128, width=512, height=512):
    for _ in range(5):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        shifted_boxes = []
        valid = True
        for box in boxes:
            x1, y1, x2, y2 = [b + s for b, s in zip(box, [shift_x, shift_y, shift_x, shift_y])]
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                valid = False
                break
            shifted_boxes.append([x1, y1, x2, y2])
        if valid:
            return shifted_boxes, classes
    return boxes, classes

def random_scale(boxes, classes, scale_range=(0.9, 1.1), width=512, height=512):
    scale = random.uniform(*scale_range)
    cx, cy = width / 2, height / 2
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - cx) * scale + cx
        x2 = (x2 - cx) * scale + cx
        y1 = (y1 - cy) * scale + cy
        y2 = (y2 - cy) * scale + cy
        scaled_boxes.append([
            np.clip(x1, 0, width) / 512,
            np.clip(y1, 0, height) / 512,
            np.clip(x2, 0, width) / 512,
            np.clip(y2, 0, height) / 512,
        ])

    return scaled_boxes, classes

def augment_data(boxes, classes):
    boxes, classes = random_flip(boxes, classes)
    boxes, classes = safe_random_shift(boxes, classes)
    boxes, classes = random_scale(boxes, classes)
    return boxes, classes

if __name__ == "__main__":
    thr = 15
    threshold = 0.0025
    height, width = 512, 512

    metadata = [eval(i) for i in open("exdark/train/metadata.jsonl", "r").readlines()]
    dataorigin = json.load(open("exdark/annotations/instances_train.json", "r"))

    categories = dataorigin["categories"]
    annotations = dataorigin["annotations"]
    images = dataorigin["images"]

    catemap = {i["id"]: i["name"] for i in categories}
    anno2image = {i["id"]: (i["width"], i["height"]) for i in images}

    datadictin = defaultdict(list)
    for item in annotations:
        dataclass = catemap[item["category_id"]]
        x, y, w, h = item["bbox"]
        width0, height0 = anno2image[item["image_id"]]
        x1, y1, x2, y2 = x, y, x + w, y + h
        box = [int(x1 / width0 * 512), int(y1 / height0 * 512),
               int(x2 / width0 * 512), int(y2 / height0 * 512)]
        datadictin[dataclass.lower()].append(box.copy())

    listin = []
    for item in metadata:
        dictin = {}
        filename = item["file_name"]
       
        caption = item["caption"][0]
        classes = [cate for cate in item["caption"][1:] if cate.strip()]
        caption_g = caption[:caption.index(",".join(classes))] if classes else caption
        bboxes = item["bndboxes"]
      
        
        cates, boxes = [], []
        for cls, box in zip(classes, bboxes):
            if sum(box) == 0:
                continue
            x1, y1, x2, y2 = [int(v * 512) for v in box]
            if (x2 - x1) * (y2 - y1) < threshold * height * width:
                continue
            boxes.append([x1, y1, x2, y2])
            cates.append(cls)
     
        boxes, cates = augment_data(boxes, cates)

        if len(cates) > thr:
            cates, boxes = cates[:thr], boxes[:thr]
        while len(cates) < thr:
            cates.append("")
            boxes.append([0, 0, 0, 0])

        r_augmented_boxes = [[i[0], i[1], i[2], i[1], i[2], i[3], i[0], i[3]] for i in boxes]
        caplist = [caption_g.strip(",").strip(".") + "," + ",".join([i for i in cates if i.strip()]).strip() + "."] + cates

        dictin["file_name"] = filename.split(".")[0] + "_gen.jpg"
        dictin["caption"] = caplist
        dictin["bndboxes"] = boxes
        dictin["obboxes"] = r_augmented_boxes
     
     
        listin.append(dictin.copy())

    with open("exdark/metadata_gen_3x.jsonl", "w", encoding="utf-8") as f:
        for item in listin:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    