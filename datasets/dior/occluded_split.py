#1.计算每张图片的目标数量和IoU(遮挡程度)；
#2.基于数据集的分布，自动确定目标数量和IoU的分界阈值；
#3.将图像归入Sparse(低遮挡)、Moderate(中等遮挡)和Dense(高遮挡)三个类别；
#4.输出每个类别的图像列表，可用于后续的模型训练或分析。

import json
import numpy as np

sparse_json = {}
moderate_json = {}
dense_json = {}

labels = json.load(open("annotations/instances_test.json","r"))

info = labels["info"]
licenses = labels["licenses"]
categories = labels["categories"]
images= labels["images"]
annotations = labels["annotations"]
total_number = len(images)


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    
    inter_area = max(0, xi2-xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def analyze_dataset(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)
        
    image_objects = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        if image_id not in image_objects:
            image_objects[image_id] = []
        image_objects[image_id].append([x1, y1, x2, y2])
        
    image_stats = []
    for image_id, bboxes in image_objects.items():
        num_objects = len(bboxes)
        iou_values = [compute_iou(b1, b2) for i, b1 in enumerate(bboxes) for j, b2 in enumerate(bboxes) if i < j]
        avg_iou = np.mean(iou_values) if iou_values else 0
        image_stats.append((image_id, num_objects, avg_iou))
        
    return image_stats

def categorize_images(annotation_file, images):
    image_stats = analyze_dataset(annotation_file)
    
    num_objects_list = [x[1] for x in image_stats]
    avg_ious_list = [x[2] for x in image_stats]
    obj_threshold1 = np.percentile(num_objects_list, 33)
    obj_threshold2 = np.percentile(num_objects_list, 66)
    iou_threshold1 = np.percentile(avg_ious_list, 33)
    iou_threshold2 = np.percentile(avg_ious_list, 66)
    
    sparse, moderate, dense = [], [], []
    for image_id, obj_count, iou in image_stats:
        if obj_count <= obj_threshold1 and iou <= iou_threshold1:
            sparse.append(image_id)
            
        elif obj_count <= obj_threshold2 and iou <= iou_threshold2:
            moderate.append(image_id)
            
        else:
            dense.append(image_id)
            
    return sparse, moderate, dense

annotation_file = "annotations/instances_test.json"

sparse, moderate, dense = categorize_images(annotation_file, images)


#sparse_json#
sparse_json["info"] = info
sparse_json["categories"] = categories
sparse_json["licenses"] = licenses
sparse_json["images"] = [item for item in images if item["id"] in sparse]
sparse_json["annotations"] = [anno for anno in annotations if anno["image_id"] in sparse]

print("Sparse (低遮挡)图像数量:",len(sparse_json["images"]), "Sparse (低遮挡)标记数量:",len(sparse_json["annotations"]))
json.dump(sparse_json,open("annotations/instances_test_sparse.json","w"))

#sparse_json#

#moderate_json#
moderate_json["info"] = info
moderate_json["categories"] = categories
moderate_json["licenses"] = licenses
moderate_json["images"] = [item for item in images if item["id"] in moderate]
moderate_json["annotations"] = [anno for anno in annotations if anno["image_id"] in moderate]

print("Moderate (中等遮挡)图像数量:",len(moderate_json["images"]), "Moderate (中等遮挡)标记数量:",len(moderate_json["annotations"]))
json.dump(moderate_json,open("annotations/instances_test_moderate.json","w"))

#moderate_json#

#dense_json#
dense_json["info"] = info
dense_json["categories"] = categories
dense_json["licenses"] = licenses
dense_json["images"] = [item for item in images if item["id"] in dense]
dense_json["annotations"] = [anno for anno in annotations if anno["image_id"] in dense]

print("dense (高遮挡)图像数量:",len(dense_json["images"]), "dense (高遮挡)标记数量:",len(dense_json["annotations"]))
json.dump(dense_json,open("annotations/instances_test_dense.json","w"))

#dense_json#

print(f"Sparse (低遮挡) 占比:{len(sparse) / total_number * 100}%, 图片ID:{sparse[:5]}")
print(f"Moderate (中等遮挡) 占比:{len(moderate) / total_number * 100}%, 图片ID:{moderate[:5]}")
print(f"Dense (高遮挡) 占比:{len(dense) / total_number * 100}%, 图片ID:{dense[:5]}")
print(f"数据集图像总数量为:{len(sparse)+len(moderate)+len(dense)}")