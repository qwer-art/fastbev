import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix, Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from utils import *
import os.path as osp


class Label(Enum):
    kHuman = 'human'
    kVehicle = 'vehicle'
    kMovable = 'movable'
    kStatic = 'static'
    kAnimal = 'animal'
    kUnknow = 'unknow'


label2color_dict = {
    Label.kHuman: Color.kGreen,
    Label.kVehicle: Color.kBlue,
    Label.kMovable: Color.kRed,
    Label.kStatic: Color.kMagenta,
    Label.kAnimal: Color.kYellow,
    Label.kUnknow: Color.kBlack
}

catname2color_dict = {
    "car": Color.kBlue,
    "truck": Color.kBlue,
    "trailer": Color.kBlue,
    "bus": Color.kBlue,
    "construction_vehicle": Color.kBlue,
    "bicycle": Color.kYellow,
    "motorcycle": Color.kYellow,
    "pedestrian": Color.kGreen,
    "traffic_cone": Color.kRed,
    "barrier": Color.kRed
}

camera_names = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

radar_names = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']


def str2label(category: str):
    ann_label = Label.kUnknow
    for label in Label:
        if category.startswith(label.value):
            return label
    return ann_label


def label2color(label: Label):
    return label2color_dict[label]


def scene_sample_tokens(nusc, scene):
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample_tokens = []
    current_sample_token = first_sample_token
    while True:
        sample_tokens.append(current_sample_token)
        if current_sample_token == last_sample_token:
            break
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    return sample_tokens


def scene_first_pose(nusc, scene):
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    tf_world_ego = sample_tf_world_ego(nusc, first_sample)
    return tf_world_ego


def scene_car_ego(nusc, scene):
    first_sample_token = scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    tf_world_ego = sample_tf_world_ego(nusc, first_sample)
    tf_ego_lidar = sample_tf_ego_lidar(nusc, first_sample)
    tf_ego_radars = sample_tf_ego_radars(nusc, first_sample)
    tf_ego_cameras = sample_tf_ego_cameras(nusc, first_sample)

    list_xyz = [tf_ego_lidar[:3, 3]]
    for tf_ego_radar in tf_ego_radars:
        list_xyz.append(tf_ego_radar[:3, 3])
    for tf_ego_camera in tf_ego_cameras:
        list_xyz.append(tf_ego_camera[:3, 3])
    list_xyz = np.array(list_xyz)
    minx = np.min(list_xyz[:, 0])
    maxx = np.max(list_xyz[:, 0])
    miny = min(np.min(list_xyz[:, 1]), -1.)
    maxy = max(np.max(list_xyz[:, 1]), 1.)
    # minz = np.min(list_xyz[:, 2])
    minz = 0
    maxz = np.max(list_xyz[:, 2])
    corners_ego = np.array([
        [maxx, maxy, minz],
        [maxx, miny, minz],
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, maxz],
        [maxx, miny, maxz],
        [minx, miny, maxz],
        [minx, maxy, maxz]
    ])
    return corners_ego


def sample_tf_world_ego(nusc, sample):
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    tf_world_key = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)
    return tf_world_key


def sample_tf_ego_lidar(nusc, sample):
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ## tf_ego_lidar
    calib_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    tf_ego_lidar = transform_matrix(calib_sensor_data['translation'], Quaternion(calib_sensor_data['rotation']),
                                    inverse=False)
    return tf_ego_lidar


def sample_tf_ego_radars(nusc, sample):
    sample_tf_ego_radars = []
    for radar_key in radar_names:
        sample_data = nusc.get('sample_data', sample['data'][radar_key])
        ## tf_ego_radar
        calib_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        tf_ego_radar = transform_matrix(calib_sensor_data['translation'], Quaternion(calib_sensor_data['rotation']),
                                        inverse=False)
        sample_tf_ego_radars.append(tf_ego_radar)
    return sample_tf_ego_radars


def sample_tf_ego_cameras(nusc, sample):
    tf_ego_cameras = []
    for camera_name in camera_names:
        sample_data = nusc.get('sample_data', sample['data'][camera_name])
        ## tf_ego_radar
        calib_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        tf_ego_camera = transform_matrix(calib_sensor_data['translation'], Quaternion(calib_sensor_data['rotation']),
                                         inverse=False)
        tf_ego_cameras.append(tf_ego_camera)
    return tf_ego_cameras


def sample_camera_intrinsics(nusc, sample):
    intrisics = []
    for camera_name in camera_names:
        sample_data = nusc.get('sample_data', sample['data'][camera_name])
        calib_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        camk = np.array(calib_sensor_data['camera_intrinsic'])
        intrisics.append(camk)
    return intrisics

def sample_lidar_cloud(nusc, sample):
    sample_data_name = 'LIDAR_TOP'
    sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
    data_path = osp.join(nusc.dataroot, sample_data['filename'])
    pointcloud = LidarPointCloud.from_file(data_path)
    return pointcloud


def sample_radar_clouds(nusc, sample):
    radar_pcs = []
    for radar_key in radar_names:
        sample_data = nusc.get('sample_data', sample['data'][radar_key])
        data_path = osp.join(nusc.dataroot, sample_data['filename'])
        radar_pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sample_data['token']))
        radar_pcs.append(radar_pc)
    return radar_pcs


def sample_images(nusc, sample):
    images = []
    for sample_data_name in camera_names:
        sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
        data_path = osp.join(nusc.dataroot, sample_data['filename'])
        image = cv2.imread(data_path)
        images.append(image)
    return images


def get_box(gt_box):
    center = gt_box[:3]
    wlh = gt_box[3:6]
    rot = - (gt_box[6] + np.pi / 2.)
    rot = - (rot + np.pi / 2.)
    q = Quaternion(axis=[0, 0, 1], radians=rot)
    return Box(center, wlh, q)

def get_bbox_edges(corners):
    edges = [
        # bottom
        (0, 1), (1, 2), (2, 3), (3, 0),
        # top
        (4, 5), (5, 6), (6, 7), (7, 4),
        # edge
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    start_points = []
    end_points = []
    for start, end in edges:
        start_points.append(corners[:, start])
        end_points.append(corners[:, end])
    return np.array(start_points), np.array(end_points)

def draw_camera():
    root_dir = osp.expanduser('~/Data/nuscenes/v1.0-mini')
    nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # region a.scene
    scene = nusc.scene[0]
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample_tokens = []
    current_sample_token = first_sample_token
    while True:
        sample_tokens.append(current_sample_token)
        if current_sample_token == last_sample_token:
            break
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    # endregion
    # region aa.sample
    sample_size = len(sample_tokens)
    sample_idx = 0
    sample = nusc.get('sample', sample_tokens[sample_idx])
    print(f"sample_size: {sample_size}")
    # endregion
    # region aaa.annotations
    anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    # endregion
    # region aaa.sample_data
    ## 1. image
    sample_data_name = 'CAM_FRONT'
    sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
    data_path = osp.join(nusc.dataroot, sample_data['filename'])
    image = cv2.imread(data_path)
    ## 2. param
    ### 2.1 ego_pose
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    e2w_p = np.array(ego_pose['translation'])
    e2w_q = Quaternion(ego_pose['rotation'])
    w2e = transform_matrix(e2w_p, e2w_q, inverse=True)
    w2e_p = w2e[:3, 3]
    w2e_q = Quaternion(matrix=w2e[:3, :3])
    ### 2.2 extrinsic
    calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    c2e_p = np.array(calib_data['translation'])
    c2e_q = Quaternion(calib_data['rotation'])
    e2c = transform_matrix(c2e_p, c2e_q, inverse=True)
    e2c_p = e2c[:3, 3]
    e2c_q = Quaternion(matrix=e2c[:3, :3])
    ### 2.3 intrinsic
    camk = np.array(calib_data['camera_intrinsic'])
    ### 2.4 project
    for idx, ann in enumerate(anns):
        #### 1.world To ego To camera
        bbox = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        bbox.rotate(w2e_q)
        bbox.translate(w2e_p)
        bbox.rotate(e2c_q)
        bbox.translate(e2c_p)
        corners_camera = bbox.corners()
        if (bbox.center[2] < 0 or not np.all(corners_camera[2, :] > 0)):
            continue
        #### 2.camera To image
        corners_image = view_points(corners_camera, camk, normalize=True)[:2, :].T
        # print("corners_camera\n",corners_camera)
        # print("camk\n",camk)
        # print("image_shape\n",image.shape)
        # print("corners_image\n",corners_image)
        # break

        #### 3.label
        bbox_label = Label.kUnknow
        for label in Label:
            if ann['category_name'].startswith(label.value):
                bbox_label = label
                break
        bgr = color2bgr(label2color(bbox_label))
        #### 4.draw
        for i in range(4):
            start_point = tuple(corners_image[i].astype(int))
            end_point = tuple(corners_image[(i + 1) % 4].astype(int))
            cv2.line(image, start_point, end_point, bgr, 3)
        for i in range(4, 8):
            start_point = tuple(corners_image[i].astype(int))
            end_point = tuple(corners_image[(i + 1) % 4 + 4].astype(int))
            cv2.line(image, start_point, end_point, bgr, 3)
        for i in range(4):
            start_point = tuple(corners_image[i].astype(int))
            end_point = tuple(corners_image[i + 4].astype(int))
            cv2.line(image, start_point, end_point, bgr, 3)
        ann_center = corners_image.mean(axis=0).astype(int)
        cv2.putText(image, str(idx), ann_center, 4, 1.0, bgr, 2)
    cv2.imshow(sample_data_name, image)
    cv2.waitKey(-1)
    # endregion

def get_sample_image(nusc, sample):
    anns = [nusc.get('sample_annotation', token) for token in sample['anns']]
    images = []
    for idx,sample_data_name in enumerate(camera_names):
        sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
        data_path = osp.join(nusc.dataroot, sample_data['filename'])
        image = cv2.imread(data_path)
        ## 2. param
        ### 2.1 ego_pose
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        e2w_p = np.array(ego_pose['translation'])
        e2w_q = Quaternion(ego_pose['rotation'])
        w2e = transform_matrix(e2w_p, e2w_q, inverse=True)
        ### 2.2 extrinsic
        calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        c2e_p = np.array(calib_data['translation'])
        c2e_q = Quaternion(calib_data['rotation'])
        e2c = transform_matrix(c2e_p, c2e_q, inverse=True)
        ### 2.3 intrinsic
        camk = np.array(calib_data['camera_intrinsic'])
        ### 2.4 project
        for idx, ann in enumerate(anns):
            #### 1.world To ego To camera
            bbox = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            oxyz = pqs2oxyz(np.array(ann['translation']), Quaternion(ann['rotation']), np.array(ann['size']), 0.5)
            oxyz_camera = transform_points(e2c @ w2e, oxyz)
            oxyz_pixel = filter_pixels(image.shape, project_points(camk, oxyz_camera))
            if oxyz_pixel.shape[0] != 4:
                continue
            cv2.arrowedLine(image, oxyz_pixel[0], oxyz_pixel[1], color2bgr(Color.kRed), 2)
            cv2.arrowedLine(image, oxyz_pixel[0], oxyz_pixel[2], color2bgr(Color.kGreen), 2)
            cv2.arrowedLine(image, oxyz_pixel[0], oxyz_pixel[3], color2bgr(Color.kBlue), 2)
            cv2.putText(image, str(idx), oxyz_pixel[0], 4, 1.2, color2bgr(label2color(str2label(ann['category_name']))),
                        3)
        cv2.putText(image, sample_data_name, (30, 80), 4, 1.5, color2bgr(Color.kGreen),
                    3)
        images.append(image)
    row1 = np.hstack(images[:3])
    row2 = np.hstack(images[3:])
    big_image = np.vstack((row1, row2))
    return big_image


def draw_cameras():
    root_dir = osp.expanduser('~/Data/nuscenes/v1.0-mini')
    nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # region a.scene
    scene = nusc.scene[0]
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample_tokens = []
    current_sample_token = first_sample_token
    while True:
        sample_tokens.append(current_sample_token)
        if current_sample_token == last_sample_token:
            break
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    # endregion
    # region aa.sample
    sample_size = len(sample_tokens)
    sample_idx = 0
    sample = nusc.get('sample', sample_tokens[sample_idx])
    print(f"sample_size: {sample_size}")
    # endregion
    sample_image = get_sample_image(nusc, sample)
    h, w, _ = sample_image.shape
    new_width = int(w / 2)
    new_height = int(h / 2)
    sample_image = cv2.resize(sample_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow("sample_image", sample_image)
    cv2.waitKey(-1)
    # endregion

def draw_lidar():
    root_dir = osp.expanduser('~/Data/nuscenes/v1.0-mini')
    nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # region a.scene
    scene = nusc.scene[0]
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample_tokens = []
    current_sample_token = first_sample_token
    while True:
        sample_tokens.append(current_sample_token)
        if current_sample_token == last_sample_token:
            break
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    # endregion
    # region aa.sample
    sample_size = len(sample_tokens)
    sample_idx = 20
    sample = nusc.get('sample', sample_tokens[sample_idx])
    print(f"sample_size: {sample_size}")
    # endregion
    # region aa.bev_map
    resolution = 0.1
    x_range = [-50, 50]
    y_range = [-50, 50]
    height = int((y_range[1] - y_range[0]) / resolution)
    width = int((x_range[1] - x_range[0]) / resolution)
    gray_value = 125
    bev_map = np.full((height, width, 3), gray_value, dtype=np.uint8)
    # endregion
    # region aaa.sample_data
    ## 1. point cloud
    sample_data_name = 'LIDAR_TOP'
    sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
    data_path = osp.join(nusc.dataroot, sample_data['filename'])
    pointcloud = LidarPointCloud.from_file(data_path)
    ## 2. param
    calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    l2e_p = np.array(calib_data['translation'])
    l2e_q = Quaternion(calib_data['rotation'])
    l2e = transform_matrix(l2e_p, l2e_q)
    ## 3. transform
    pointcloud.transform(l2e)
    ## 4. draw
    x_indices = np.floor((pointcloud.points[0, :] - x_range[0]) / resolution).astype(int)
    y_indices = np.floor((pointcloud.points[1, :] - y_range[0]) / resolution).astype(int)
    valid_indices = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
    bev_map[y_indices[valid_indices], x_indices[valid_indices]] = color2bgr(Color.kBlue)
    txt_idx = 0
    txt_idx = txt_idx + 1
    txt = "cloud: " + str(pointcloud.points.shape[1])
    cv2.putText(bev_map, txt, (30, txt_idx * 60), 4, 1.5, color2bgr(Color.kBlue), 2)

    cv2.imshow("bev_map", bev_map)
    cv2.waitKey(-1)
    # endregion


def draw_radar():
    root_dir = osp.expanduser('~/Data/nuscenes/v1.0-mini')
    nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # region a.scene
    scene = nusc.scene[0]
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample_tokens = []
    current_sample_token = first_sample_token
    while True:
        sample_tokens.append(current_sample_token)
        if current_sample_token == last_sample_token:
            break
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    # endregion
    # region aa.sample
    sample_size = len(sample_tokens)
    sample_idx = 20
    sample = nusc.get('sample', sample_tokens[sample_idx])
    print(f"sample_size: {sample_size}")
    # endregion
    # region aa.bev_map
    resolution = 0.1
    x_range = [-50, 50]
    y_range = [-50, 50]
    height = int((y_range[1] - y_range[0]) / resolution)
    width = int((x_range[1] - x_range[0]) / resolution)
    gray_value = 125
    bev_map = np.full((height, width, 3), gray_value, dtype=np.uint8)
    # endregion
    # region aaa.sample_data
    ## 1. point cloud
    sample_data_name = 'RADAR_FRONT'
    sample_data = nusc.get('sample_data', sample['data'][sample_data_name])
    data_path = osp.join(nusc.dataroot, sample_data['filename'])
    radar_pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sample_data['token']))
    ## 2. param
    calib_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    r2e_p = np.array(calib_data['translation'])
    r2e_q = Quaternion(calib_data['rotation'])
    r2e = transform_matrix(r2e_p, r2e_q)
    ## 3. transform
    radar_pc.transform(r2e)
    ## 4. draw
    x_indices = np.floor((radar_pc.points[0, :] - x_range[0]) / resolution).astype(int)
    y_indices = np.floor((radar_pc.points[1, :] - y_range[0]) / resolution).astype(int)
    valid_indices = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
    for valid, u, v in zip(valid_indices, x_indices, y_indices):
        if not valid:
            continue
        cv2.circle(bev_map, (u, v), 3, color2bgr(Color.kRed), -1)
    # bev_map[y_indices[valid_indices], x_indices[valid_indices]] = color2bgr(Color.kRed)
    txt_idx = 0
    txt_idx = txt_idx + 1
    txt = "cloud: " + str(radar_pc.points.shape[1])
    cv2.putText(bev_map, txt, (30, txt_idx * 60), 4, 1.5, color2bgr(Color.kRed), 2)

    cv2.imshow("bev_map", bev_map)
    cv2.waitKey(-1)
    # endregion

if __name__ == '__main__':
    draw_cameras()