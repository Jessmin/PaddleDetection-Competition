# 输入dataset 路径转换为VOC格式
import os
import shutil
import glob
import random

DATASET_BASE_DIR = '/home/data/VOC'
JPEGIMAGES_DIR = f'{DATASET_BASE_DIR}/JPEGImages'
ANNO_DIR = f'{DATASET_BASE_DIR}/Annotations'
os.makedirs(DATASET_BASE_DIR, exist_ok=True)
os.makedirs(JPEGIMAGES_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)

jpgs = glob.glob('/home/data/*/*.jpg')
jpgs.extend(glob.glob('/home/data/*.jpg'))

xmls = glob.glob('/home/data/*/*.xml')
xmls.extend(glob.glob('/home/data/*.xml'))
for item in jpgs:
    _, fname = os.path.split(item)
    dst = os.path.join(JPEGIMAGES_DIR, fname)
    shutil.copy(item, dst)
for item in xmls:
    _, fname = os.path.split(item)
    dst = os.path.join(JPEGIMAGES_DIR, fname)
    shutil.copy(item, dst)
# convert dataset
imgs = os.listdir(JPEGIMAGES_DIR)
imgs.sort()
f_val = open(f'{DATASET_BASE_DIR}/val.txt', 'w')
with open(f'{DATASET_BASE_DIR}/train.txt', 'w') as f:
    for idx, item in enumerate(imgs):
        if random.randint(0, 100) < 90:
            f.write(f"{item.split('.')[0]}\n")
        else:
            f_val.write(f"{item.split('.')[0]}\n")

label_list = ['slagcar', 'car', 'tricar', 'motorbike', 'bicycle', 'bus', 'truck', 'tractor']
with open(f'{DATASET_BASE_DIR}/label.txt', 'w') as f:
    for label in label_list:
        f.write(f"{label}\n")
