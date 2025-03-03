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
import matplotlib.pyplot as plt

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def sample_fvimgs_2_image(fvimgs):
    sample_images = []
    for img in fvimgs:
        img = (img - img.min()) / (img.max() - img.min())
        img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype(np.uint8)
        sample_images.append(img)
    row1 = np.hstack(sample_images[:3])
    row2 = np.hstack(sample_images[3:])
    sample_image = np.vstack((row1, row2))
    return sample_image

def get_sample_fv(dataset,batch_idx,sample_idx):
    batch_data = dataset[batch_idx]
    batch_image_datas = batch_data['img'].data

    sample_size = 4
    image_size = 6
    N,C,H,W = batch_image_datas.shape
    batch_image_datas = batch_image_datas.view(sample_size,image_size,C,H,W)
    sample_data = batch_image_datas[sample_idx]
    sample_image = sample_fvimgs_2_image(sample_data)
    return sample_image

def bev_image_2_visual_images(bev_image):
    bev_imgs = []
    gray_tensor_1 = bev_image[..., 0]
    gray_tensor_2 = bev_image[..., 1]

    gray_array_1 = gray_tensor_1.numpy()
    gray_array_2 = gray_tensor_2.numpy()

    gray_array_1 = cv2.normalize(gray_array_1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_array_2 = cv2.normalize(gray_array_2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    color_image_1 = cv2.cvtColor(gray_array_1, cv2.COLOR_GRAY2BGR)
    color_image_2 = cv2.cvtColor(gray_array_2, cv2.COLOR_GRAY2BGR)

    return [color_image_1,color_image_2]

def debug():
    ### 1.input
    config_path = '/home/jerett/Project/Fast-BEV/configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.train)
    batch_size = len(dataset)
    sample_size = 4
    image_size = 6

    ### 2.fv_image
    sample_image = get_sample_fv(dataset, 1, 2)
    cv2.imshow("sample_data", sample_image)
    cv2.waitKey(-1)

    # fv_img_list = fv_image_2_visual_images(fv_images)
    # print(f"fv_image: {fv_images.shape},{len(fv_img_list)},{fv_img_list[0].shape}")
    # ## 3.2 process bev
    # bev_images= batch_data['gt_bev_seg'].data
    # bev_img_list = bev_image_2_visual_images(bev_images)
    # print(f"bev_images: {bev_images.shape},{len(bev_img_list)},{bev_img_list[0].shape}")

def main():
    # region input
    config_path = '/home/jerett/Project/Fast-BEV/configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.train)

    batch_size = len(dataset)
    sample_size = 4
    image_size = 6
    N, C, H, W = 24, 3, 256, 704
    center = np.array([0, 0, 0])
    #endregion

    # region param
    screen_w = 1920  # 假设屏幕宽度
    screen_h = 1080  # 假设屏幕高度
    try:
        output = subprocess.check_output(['xrandr']).decode('utf-8')
        for line in output.splitlines():
            if '*' in line:
                screen_w, screen_h = map(int, line.split()[0].split('x'))
                break
    except Exception as e:
        screen_w = 1920
        screen_h = 1080

    image_w = W * 3
    image_h = H * 2

    pangolin.CreateWindowAndBind('Main', screen_w, screen_h)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(screen_w, screen_h, 2000, 2000, 960, 540, 0.1, 200),
        pangolin.ModelViewLookAt(center[0], center[1], center[2] + 70, center[0], center[1], center[2], 1, 0, 0))
    handler = pangolin.Handler3D(scam)

    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -screen_w / screen_h)
    dcam.SetHandler(handler)

    dimg = pangolin.Display('image')
    dimg.SetBounds(0.5, 1.0, 1. / 6., 1.0, float(image_w) / float(image_h))
    dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
    texture = pangolin.GlTexture(image_w, image_h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    # endregion

    # region panel
    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(0.0, 1., 0.0, 1. / 6.)
    auto_play = pangolin.VarBool('ui.AutoPlay', value=False, toggle=False)
    play_step = pangolin.VarBool('ui.>>', value=False, toggle=False)
    play_back = pangolin.VarBool('ui.<<', value=False, toggle=False)
    curr_batch_idx = pangolin.VarInt('ui.batch_idx', value=0, min=0, max=batch_size - 1)
    play_sample_step = pangolin.VarBool('ui.sp>>', value=False, toggle=False)
    play_sample_back = pangolin.VarBool('ui.sp<<', value=False, toggle=False)
    curr_sample_idx = pangolin.VarInt('ui.sample_idx', value=0, min=0, max=image_size - 1)
    show_grid = pangolin.VarBool('ui.grid', value=True, toggle=True)
    show_fv_image = pangolin.VarBool('ui.fv_image', value=True, toggle=True)
    # endregion

    frequency = 10.
    last_time = time.time() * 1000
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gray_color = 125. / 255.
        gl.glClearColor(gray_color, gray_color, gray_color, 1.0)
        dcam.Activate(scam)

        curr_time = time.time() * 1000
        dt_ms = curr_time - last_time

        batch_idx = curr_batch_idx.Get()
        if auto_play.Get() or play_step.Get():
            play_step.SetVal(False)
            if dt_ms > (1000. / frequency):
                last_time = curr_time
                batch_idx = batch_idx + 1
            batch_idx = batch_idx % batch_size

        if play_back.Get():
            play_back.SetVal(False)
            batch_idx = batch_idx - 1
            if batch_idx < 0:
                batch_idx = batch_size - 1
        curr_batch_idx.SetVal(batch_idx)

        sample_idx = curr_sample_idx.Get()
        if play_sample_step.Get():
            play_sample_step.SetVal(False)
            sample_idx = sample_idx + 1
            sample_idx = sample_idx % sample_size
        if play_sample_back.Get():
            play_sample_back.SetVal(False)
            sample_idx = sample_idx - 1
            if sample_idx < 0:
                sample_idx = sample_size - 1
        curr_sample_idx.SetVal(sample_idx)

        #region sample_data
        batch_data = dataset[batch_idx]
        batch_image_datas = batch_data['img'].data

        batch_image_datas = batch_image_datas.view(sample_size, image_size, C, H, W)
        sample_data = batch_image_datas[sample_idx]
        #endregion

        if show_grid.Get():
            draw_grid(10., center)

        if show_fv_image.Get():
            sample_image = get_sample_fv(dataset, batch_idx, sample_idx)
            image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            texture.Upload(image_rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            dimg.Activate()
            gl.glColor3f(1.0, 1.0, 1.0)
            texture.RenderToViewportFlipY()

        pangolin.FinishFrame()


if __name__ == '__main__':
    # debug()
    main()