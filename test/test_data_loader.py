import os.path as osp
import sys
import json
project_path = osp.dirname(osp.dirname(__file__))
sys.path.append(project_path)
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
import argparse
import pickle
import numpy as np

def config_python_to_config_json():
    config = osp.join(project_path, "configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py")
    cfg = Config.fromfile(config)

    # 将 ConfigDict 转换为字典
    cfg_dict = cfg.to_dict()

    # 保存为 JSON 文件
    save_path = osp.join(project_path, "test_dir/config_python.json")
    print(f"save_config_json: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(cfg_dict, f, indent=4)

def create_data_cfg_to_json():
    parser = argparse.ArgumentParser(description='create data parser')
    parser.add_argument(
        '--dataset',
        type=str,
        default='nuscenes',
        help='dataset')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/nuscenes',
        help='name of info pkl')
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    parser.add_argument(
        '--workers', type=int, default=10, help='number of threads to be used')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        required=False,
        help='specify the dataset version, no need for kitti')
    args = parser.parse_args()# 获取所有参数名和值
    params = vars(args)
    save_path = osp.join(project_path,"test_dir/create_data_parser.json")
    print(f"save_config_json: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)

def train_configs_to_json():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py',
        help='dataset')

    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dir',
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=0,
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--wo-wandb',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--wo-fp16',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    params = vars(args)
    save_path = osp.join(project_path, "test_dir/train_parser.json")
    print(f"save_config_json: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)

def convert_to_serializable(obj):
    # 递归将数据中的非JSON可序列化对象转换为可序列化格式
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 如果包含NumPy数组，将其转换为列表
    elif isinstance(obj, np.generic):
        return obj.item()  # 处理 numpy 标量类型（如 int32、float32 等），转换为原生 Python 类型
    else:
        return obj  # 返回原始对象

def train_infos_to_json():
    ### train_dbinfos
    data_path = osp.join(project_path, "data/nuscenes/nuscenes_dbinfos_train.pkl")
    data = pickle.load(open(data_path, "rb"))
    save_path = osp.join(project_path, "test_dir/nuscenes_dbinfos_train.json")
    serializable_data = convert_to_serializable(data)

    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    ### train_infos
    data_path = osp.join(project_path, "data/nuscenes/nuscenes_infos_train.pkl")
    data = pickle.load(open(data_path, "rb"))
    save_path = osp.join(project_path, "test_dir/nuscenes_infos_train.json")
    serializable_data = convert_to_serializable(data)
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    ### train_4d_infos
    data_path = osp.join(project_path, "data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl")
    data = pickle.load(open(data_path, "rb"))
    save_path = osp.join(project_path, "test_dir/nuscenes_infos_train_4d_interval3_max60.json")
    serializable_data = convert_to_serializable(data)
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

if __name__ == '__main__':
    train_infos_to_json()