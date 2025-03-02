import os
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mmcv.parallel.data_container import DataContainer
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
import re
import torch
import time
from torch.utils.data import DataLoader
from mmcv.parallel import collate
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.apis import train_model
import torch.nn as nn
from mmseg.ops import resize

def main():
    config_path = '/home/jerett/Project/Fast-BEV/configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'
    cfg = Config.fromfile(config_path)

    ## 1. dataset
    dataset = build_dataset(cfg.data.train)
    batch_data = dataset[0]
    img = batch_data['img'].data.unsqueeze(0)
    img_metas = [batch_data['img_metas'].data]

    ## 2.reshape
    batch_size = img.shape[0]
    img = img.reshape(
        [-1] + list(img.shape)[2:]
    )

    ## 2.backbone
    backbone = build_backbone(cfg.model.backbone)
    x = backbone(img)
    print(f"backbone,x: {type(x)}")
    for idx,item in enumerate(x):
        print(f"backbone,{idx}: {item.shape}")

    ## 3.neck
    neck = build_neck(cfg.model.neck)
    mlvl_feats = neck(x)
    mlvl_feats = list(mlvl_feats)
    print(f"neck,mlvl_feats: {type(mlvl_feats)}")
    for idx, item in enumerate(mlvl_feats):
        print(f"neck,{idx}: {item.shape}")

    ## 4.neck fuse
    multi_scale_id = [1]
    neck_fuse_0 = nn.Conv2d(256, 64, 3, 1, 1)
    mlvl_feats_ = []
    for msid in multi_scale_id:
        fuse_feats = [mlvl_feats[msid]]
        for i in range(msid + 1, len(mlvl_feats)):
            resized_feat = resize(
                mlvl_feats[i],
                size=mlvl_feats[msid].size()[2:],
                mode="bilinear",
                align_corners=False)
            fuse_feats.append(resized_feat)
        if len(fuse_feats) > 1:
            fuse_feats = torch.cat(fuse_feats, dim=1)
        else:
            fuse_feats = fuse_feats[0]
        # fuse_feats = neck_fuse_0(fuse_feats)
        # mlvl_feats_.append(fuse_feats)
    mlvl_feats = mlvl_feats_
    print(f"fuse_feats: {fuse_feats.shape}")
    # print(f"mlvl_feature: {len(mlvl_feats)},mlvl_feature0: {mlvl_feats[0].shape}")

    print("hello world")


if __name__ == "__main__":
    # calculate_loss_once()
    main()