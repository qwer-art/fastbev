import cv2
import mmcv
import numpy as np
import pangolin
from mmcv import Config, DictAction

from utils_pangl import *
from utils_nuscene import *
import os
import subprocess
import math
import time
import torch
from mmdet3d.datasets import build_dataset
import torchvision.transforms.functional as F
from PIL import Image

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def batch_data_2_sample_images(img_data):
    ## 1.split_data
    N,C,H,W = img_data.shape
    img_data = img_data.view(4, 6, C, H, W)
    ## 2.concat
    batch_sample_images = []
    for simg in img_data:
        sample_images = []
        for img in simg:
            img = (img - img.min()) / (img.max() - img.min())
            img = img.permute(1,2,0)
            sample_images.append(img.numpy())
        row1 = np.hstack(sample_images[:3])
        row2 = np.hstack(sample_images[3:])
        sample_image = np.vstack((row1, row2))
        batch_sample_images.append(sample_image)
    return batch_sample_images

def debug():
    ### 1.input
    config_path = '/home/jerett/Project/Fast-BEV/configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.train)
    batch_data = dataset[0]
    ### 2.test
    print(f"dataset: {len(dataset)}")
    print(f"batch_data: {batch_data.keys()}")

    print(f"img: {batch_data['img'].data.shape}")

    img_data = batch_data['img'].data
    batch_sample_images = batch_data_2_sample_images(img_data)

    img_metas = batch_data['img_metas'].data
    print(f"img_metas: {type(img_metas)},{img_metas.keys()}")
    print(f"filename: {img_metas['filename']}")
    print(f"img_info: {len(img_metas['img_info'])}")

    # for idx, image in enumerate(batch_sample_images):
    #     cv2.imshow(f"image_{idx}", image)
    # cv2.waitKey(-1)



if __name__ == '__main__':
    debug()