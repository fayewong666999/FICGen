import numpy as np

import cv2
from itertools import combinations
def compute_overlap_box(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    if x1 < x2 and y1 < y2:
        return [x1, y1, x2, y2]
    return None

def to_pixel(box, H, W):
    x1 = int(box[0] * W)
    y1 = int(box[1] * H)
    x2 = int(box[2] * W)
    y2 = int(box[3] * H)
    return [x1, y1, x2, y2]

def create_masks_with_overlap(boxes, labels, H=512, W=512):
    masks = {}
    new_boxes = []
    new_labels = []
    N = len(boxes)
    instance_labels = []
    label_count = {}
    
    for label in labels:
        label_count[label] = label_count.get(label, 0) + 1
    label_index = {}
    for idx, label in enumerate(labels):
        label_index[label] = label_index.get(label, 0)
        instance_label = f"{label}_{label_index[label]}"
        instance_labels.append(instance_label)
        label_index[label] += 1
    original_masks = []
    
    for i, box in enumerate(boxes):
        mask = np.zeros((H,W),dtype=np.uint8)
        x1, y1, x2, y2 = to_pixel(box, H, W)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1,thickness=-1)
        original_masks.append(mask.copy())
        
    final_masks = [mask.copy() for mask in original_masks]
    
    for (i, boxA), (j, boxB) in combinations(enumerate(boxes),2):
       
        overlap = compute_overlap_box(boxA, boxB)
        if overlap:
            px_overlap = to_pixel(overlap, H, W)
            x1, y1, x2, y2 = px_overlap
            final_masks[i][y1:y2+1,x1:x2+1]=0
            final_masks[j][y1:y2+1,x1:x2+1]=0
            overlap_mask = np.zeros((H,W),dtype=np.uint8)
            cv2.rectangle(overlap_mask,(x1,y1),(x2,y2),color=1,thickness=-1)
            new_boxes.append(overlap)
            new_label = f"{instance_labels[i]} {instance_labels[j]}"
            new_labels.append(new_label)
            masks[new_label] = overlap_mask
            
    for i,mask in enumerate(final_masks):
        label = instance_labels[i]
        masks[label] = mask
        new_boxes.append(boxes[i])
        new_labels.append(label)
        
    return masks, new_boxes, new_labels
'''

boxes = [[0.3625, 0.06, 0.4875, 0.925], [0.2025, 0.08625, 0.225, 0.17125], [0.30125, 0.20875, 0.31375, 0.23625], [0.53625, 0.08875, 0.69, 0.86125], [0.72875, 0.0875, 0.84375, 0.8525], [0.87875, 0.08125, 0.98125, 0.845], [0.5025, 0.0975, 0.53, 0.14625], [0.415, 0.79875, 0.43625, 0.86375], [0.42, 0.8725, 0.4625, 0.88625], [0.41125, 0.9025, 0.48125, 0.92125], [0.415, 0.73125, 0.4325, 0.78125], [0.435, 0.7225, 0.445, 0.74875], [0.41625, 0.64, 0.44, 0.66875], [0.415, 0.68, 0.44, 0.70875], [0.37, 0.54125, 0.395, 0.63]]]

labels = ['expressway-toll-station', 'vehicle', 'vehicle', 'vehicle']

masks, all_boxes, all_labels = create_masks_with_overlap(boxes, labels, H=512, W=512)
for label,mask in masks.items():
    filename = label.replace(" ","_") + "_mask.png"
    cv2.imwrite(filename, mask*255)
    
print("所有box（包括重叠区域）：")
for box, label in zip(all_boxes, all_labels):
    print(label, box)
    
'''
    
    
    
    