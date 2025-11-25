import os
import json
import cv2
import random
import time
from PIL import Image
import xml.etree.ElementTree as ET


def anno_parser(anno_path, dataset_name="DIOR"):
    name_list = anno_path["caption"][1:]
    name_list = [i for i in name_list if i.strip()]
    bbox_list = anno_path["bndboxes"]
    bbox_list = [i for i in bbox_list if sum(i)!=0]
        
    return name_list, bbox_list
   

    
coco_format_save_path='/cpfs/user/wenzhuangwang/FICGen/datasets/dior/annotations/'   #要生成的标准coco格式标签所在文件夹
yolo_format_classes_path='/cpfs/user/wenzhuangwang/FICGen/datasets/dior/names.txt'     #类别文件，一行一个类
yolo_format_annotation_path='/cpfs/user/wenzhuangwang/FICGen/datasets/dior/labels/'  #yolo格式标签所在文件夹
img_pathDir='/cpfs/user/wenzhuangwang/FICGen/datasets/dior/train/'    #图片所在文件夹

with open(yolo_format_classes_path,'r') as fr:                               #打开并读取类别文件
    lines1=fr.readlines()
lines1 = [i.strip() for i in lines1 if i.strip()]
# print(lines1)
categories=[]                                                                 #存储类别的列表
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({'id':j+1,'name':label,'supercategory':'None'})         #将类别信息添加到categories中
# print(categories)


import json
annos = json.load(open(os.path.join(coco_format_save_path, "instances_train.json")))

original_images = annos["images"]
original_annos = annos["annotations"]
write_json_context=dict()                                                      #写入.json文件的大字典
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '纯粹ss', 'date_created': '2022-07-8'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']= original_images[:]
write_json_context['annotations']= original_annos[:]

num_images = len(original_images)
num_annos = len(original_annos)

print(num_images,num_annos)
gt_map = {}

#接下来的代码主要添加'images'和'annotations'的key值
imageFileList=open("dior/metadata_gen.jsonl","r").readlines()
for item in imageFileList:
    item = eval(item)
    gt_map[item["file_name"]] = item
imageFileList = [eval(i)["file_name"] for i in imageFileList]
imageFileList = [i for i in imageFileList if i.endswith("jpg")]
print(len(imageFileList))
#遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径
#    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
#    W, H = 512,512
 
    W, H = Image.open(os.path.join(img_pathDir, imageFile.replace("_gen", ""))).convert("RGB").size

    
    img_context={}                                                              #使用一个字典存储该图片信息
    #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name']=imageFile.split(".")[0]+".jpg"
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2022-07-8'
    img_context['id']= i +  num_images + 1                                          #该图片的id
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中
                                                 #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
    anno_path = gt_map[imageFile]
    obj_name_list, bbox_list = anno_parser(anno_path)

    for j, (class_id, line) in enumerate(zip(obj_name_list, bbox_list)):

        bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))

        x1,y1,x2,y2=line                                          #获取每一个标注框的详细信息
        class_id,x1, y1, x2, y2 = int(lines1.index(class_id.lower())), float(x1), float(y1), float(x2), float(y2)       #将字符串类型转为可计算的int和float类型

        xmin=int(x1 * W)                                                             #坐标转换
        ymin=int(y1 * H)
        xmax=int(x2 * W)
        ymax=int(y2 * H)
        w = xmax-xmin 
        h = ymax-ymin

        bbox_dict['id']=(i +  num_images + 1) * 1000+j + num_annos + 1                                                         #bounding box的坐标信息
        bbox_dict['image_id']=i +  num_images + 1 
        bbox_dict['category_id']=class_id + 1                                              #注意目标类别要加一
        bbox_dict['iscrowd']=0
        height,width=abs(ymax-ymin),abs(xmax-xmin)
        bbox_dict['area']=height*width
        bbox_dict['bbox']=[xmin,ymin,w,h]
        bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
        write_json_context['annotations'].append(bbox_dict)  
        

print(len(write_json_context['images']))
        #将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path,"instances_train_mix"+ '.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(write_json_context,fw,indent=2)

