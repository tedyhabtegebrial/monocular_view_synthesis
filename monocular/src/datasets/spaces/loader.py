import os
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .utils import ReadScene as read_scene
from PIL import Image
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize


seed = 42
class SpacesLoader(Dataset):
    def __init__(self, configs):
        super(SpacesLoader, self).__init__()
        self.configs = configs
        self.data_path = configs['dataset_root']
        folders_list = []
        self.mode = configs['mode']
        if self.mode=='train':
            folders_list.extend(self.get_folders('800'))
            folders_list.extend(self.get_folders('2k'))
        else:
            folders_list.extend(self.get_folders('eval'))
        self.folders_list = folders_list
        self.rng = np.random.RandomState(seed)
        self.transform = Compose([
								Resize((self.configs['width'], self.configs['height'])),
								ToTensor(),
								# Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
								])


    def get_folders(self, folder_path):
        data_path = self.data_path
        files_1 = os.listdir(os.path.join(data_path, folder_path))
        files_1 = [os.path.join(data_path, folder_path, f) for f in files_1]
        files_1 = [f for f in files_1 if os.path.isdir(f)]
        return files_1

    def __len__(self):
        return len(self.folders_list)

    def __getitem__(self,index):
        current_folder = self.folders_list[index]
        scene_views = read_scene(current_folder)
        scene_len = len(scene_views)
        src_rig = self.rng.randint(0, scene_len)
        target_rig = self.rng.randint(1, self.configs['max_baseline']) + src_rig
        target_rig = max(min(target_rig, scene_len - 1), 0)

        src_camera_idx = self.rng.randint(0, len(scene_views[src_rig]))
        target_camera_idx = self.rng.randint(0, len(scene_views[src_rig]))
        src_view = scene_views[src_rig][src_camera_idx]
        target_view = scene_views[target_rig][target_camera_idx]

        src_img = Image.open(src_view.image_path)
        target_img = Image.open(target_view.image_path)
        data_dict = {}
        data_dict['input_img'] = self.transform(src_img) * 2.0 - 1.0
        data_dict['target_img'] = self.transform(target_img) * 2.0 - 1.0
        # print(src_view.image_path)
        # print(data_dict['input_img'].shape)
        # print(target_view.image_path)
        # print(data_dict['target_img'].shape)
        data_dict['diff_kmats'] = True
        data_dict['k_mats'] = torch.Tensor(src_view.camera.intrinsics)
        data_dict['k_mats_target'] = torch.Tensor(target_view.camera.intrinsics)
        pose_src = src_view.camera.c_f_w
        pose_target = target_view.camera.w_f_c
        r_src, t_src = torch.Tensor(pose_src[:3, :3]), torch.Tensor(pose_src[:3, 3])
        r_target, t_target = torch.Tensor(pose_target[:3, :3]), torch.Tensor(pose_target[:3, 3])
        r_rel = torch.mm(r_target, r_src)
        t_rel = torch.mm(r_target, (t_src - t_target).view(3, 1))
        data_dict['r_mats'] = r_rel
        data_dict['t_vecs'] = t_rel
        # for k in data_dict:
        #     print(k)
        #     print(data_dict[k])
        return data_dict
        # each scene contains the following
        # for view in scene_views:
        #     print(len(scene_views))
        #     print(view[0].image_path)
        #     print(view[0].camera.intrinsics)
        #     break


if __name__=='__main__':
    configs = {}
    configs['dataset_root'] = '/home5/anwar/data/spaces/spaces_dataset/data'
    configs['mode'] = 'train'
    configs['max_baseline'] = 5
    configs['height'] = 384
    configs['width'] = 256

    loader = SpacesLoader(configs, 'train')
    loader.__getitem__(10)
    loader.__getitem__(10)
    loader.__getitem__(10)
