from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import os
import os.path as osp
import mmcv
from pycodestyle import break_before_binary_operator
from pyquaternion import Quaternion
import numpy as np
from mmdet3d.datasets import NuScenesDataset
from utils import *
from nuscenes.utils.geometry_utils import view_points
from typing import List, Tuple, Union
from collections import OrderedDict
from mmdet3d.core.bbox.box_np_ops import points_cam2img

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    ### transform
    tf_ego_lidar = pr2tf((l2e_t,l2e_r_mat))
    tf_lidar_ego = np.linalg.inv(tf_ego_lidar)
    tf_global_ego = pr2tf((e2g_t,e2g_r_mat))
    tf_ego_global = np.linalg.inv(tf_global_ego)
    tf_egos_sensor = pq2tf((l2e_t_s,l2e_r_s))
    tf_global_egos = pq2tf((e2g_t_s,e2g_r_s))

    tf_lidar_sensor = tf_lidar_ego @ tf_ego_global @ tf_global_egos @ tf_egos_sensor
    sweep['sensor2lidar_rotation'] = tf_lidar_sensor[:3,:3]
    sweep['sensor2lidar_translation'] = tf_lidar_sensor[:3,3]

    return sweep

### 1.[3d]
# input: nuscenes
# output: nuscenes_infos_train.pkl,list(sample)
def debug_sample_infos():
    ## input
    root_path = './data/nuscenes'
    info_prefix = 'nuscenes'
    version = 'v1.0-mini'
    dataset_name = 'NuScenesDataset'
    out_dir = 'debug'
    max_sweeps = 10

    ## 1.split train/val
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    split_idx = 8
    train_scenes_token = set(scene['token'] for scene in nusc.scene[:split_idx])
    val_scenes_token = set(scene['token'] for scene in nusc.scene[split_idx:])

    base_time = nusc.sample[0]['timestamp']

    for idx,sample in enumerate(nusc.sample):
        print(f"============= {idx}: {(sample['timestamp'] - base_time) / 1e6} =============")
        if idx >= 10:
            break

        #region 1.train_flag
        train_flag = True
        if sample['scene_token'] not in train_scenes_token:
            train_flag = False
        #endregion

        #region 2.lidar
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        tf_ego_lidar = pq2tf((l2e_t,l2e_r))

        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        tf_global_ego = pq2tf((e2g_t,e2g_r))

        l2e_r_mat = tf_ego_lidar[:3,:3]
        e2g_r_mat = tf_global_ego[:3,:3]
        #endregion

        #region 3.camera
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})
        #endregion

        #region 4. sweeps
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # region 5.ann
        annotations = [
            nusc.get('sample_annotation', token)
            for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        velocity = np.array(
            [nusc.box_velocity(token)[:2] for token in sample['anns']])
        valid_flag = np.array(
            [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
             for anno in annotations],
            dtype=bool).reshape(-1)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NuScenesDataset.NameMapping:
                names[i] = NuScenesDataset.NameMapping[names[i]]
        names = np.array(names)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array(
            [a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array(
            [a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = valid_flag
        #endregion


if __name__ == '__main__':
    debug_sample_infos()
