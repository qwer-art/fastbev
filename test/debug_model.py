# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

from mmcv import Config, DictAction
from mmcv.runner.optimizer.builder import build_optimizer
from mmcv.runner import build_runner
import os
import os.path as osp
import sys
project_path = osp.dirname(osp.dirname(__file__))
sys.path.append(project_path)

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.utils import get_root_logger

def get_data_loader(cfg):
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=None,
        shuffle=cfg.get('shuffle', True))
    return dataloader

def get_model(cfg):
    cfg = Config.fromfile(config_path)
    ### model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model

if __name__ == '__main__':
    config_path = "configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py"
    config_path = osp.join(project_path, config_path)
    cfg = Config.fromfile(config_path)
    ### 1.dataloader
    dataloader = get_data_loader(cfg)
    ### 2.model
    model = get_model(cfg)

    backbone = model.backbone
    print(f"backbone: {type(backbone)}")
