# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import os
import os.path as osp
import sys

import torch

project_path = osp.dirname(osp.dirname(__file__))
sys.path.append(project_path)
from mmcv import Config, DictAction
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.parallel.data_container import DataContainer
import numpy as np
mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    sys.path.insert(0, mmdet3d_root)

def print_data_type(data: DataContainer):
    if type(data.data) is list:
        print(f"a.list_data: {len(data.data)}")
    elif type(data.data) is dict:
        print(f"b.dict_data: {len(data.data)}")
    elif type(data.data) is np.ndarray:
        print(f"c.array_data: {data.data.shape}")
    elif type(data.data) is torch.Tensor:
        print(f"d.tensor_data: {data.data.shape}")
    elif type(data.data) is LiDARInstance3DBoxes:
        print(f"e.box3d: {data.data.corners.shape}")
    else:
        print("unknow type")

if __name__ == '__main__':
    config_path = "configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py"
    config_path = osp.join(project_path, config_path)
    cfg = Config.fromfile(config_path)
    print("============= dataset =============")
    dataset = build_dataset(cfg.data.train)
    print(f"[1] dataset: {len(dataset)}")
    idx = 0
    first_data = dataset[idx]
    print(f"first_data: {type(first_data)}")

    for k,v in first_data.items():
        print(f"==== key: {k} ====")
        print_data_type(v)

    print("============= dataloader =============")
    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=None,
        shuffle=cfg.get('shuffle', True))

    for batch_data in data_loader:
        for k, v in batch_data.items():
            print(f"==== key: {k},type: {type(v)} ====")
            print_data_type(v)
        break
