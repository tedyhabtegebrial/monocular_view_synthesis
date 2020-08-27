"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import os
import random
import pathlib
from pathlib import Path

from PIL import Image
import cv2 as cv
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .kitti_utils import resize_segmentation_maps, label_to_one_hot
# from .city_scapes_labels import label2trainid, trainId2label

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

class KittiLoader(Dataset):
    def __init__(self, configs):
        super(KittiLoader, self).__init__()
        base_path = configs['dataset_root']
        # base_path = '/data/Datasets/KittiOdometry/dataset'
        self.configs = configs
        self.base_path = base_path
        # select training and test sequences
        test_seqs = ['09', '10']
        train_seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
        # collect all potential target views
        self.train_frames = self._collect_frames(base_path, train_seqs, configs['max_baseline'])
        self.test_frames = self._collect_frames(base_path, test_seqs, configs['max_baseline'])

        # lets read kmatrices of each sequence and save them in a list
        self.train_kmats = self._collect_kmats(train_seqs)
        self.test_kmats = self._collect_kmats(test_seqs)

        # lets keep list of camera poses
        self.train_poses = self._collect_poses(train_seqs)
        self.test_poses = self._collect_poses(test_seqs)

    def _collect_frames(self, base_path, sequences, max_baseline):
        frame_list = []
        for seq in sequences:
            seq_path = Path(base_path, 'sequences', seq, 'image_2')
            files = sorted([str(f) for f in seq_path.rglob('*.png')])
            frame_list.extend(files[max_baseline:-max_baseline])
        return frame_list

    def _collect_poses(self, sequences):
        poses = {}
        for seq in sequences:
            pose_path = Path(self.base_path, 'poses', f'{seq}.txt')
            pose_data = np.loadtxt(str(pose_path))
            pose_data = np.reshape(pose_data, (pose_data.shape[0], 3, 4))
            poses[seq] = pose_data
        return poses

    def _collect_kmats(self, seq_list):
        list_of_kmats = {}
        for seq in seq_list:
            calib_file = Path(self.base_path, 'sequences', seq, 'calib.txt')
            with open(calib_file, 'r') as fid:
                lines = []
                for line in fid.readlines():
                    lines.append(list(map(float, line.split()[1:])))
                kmat = torch.Tensor(lines[2]).view(3,4)[0:3,0:3]
            list_of_kmats[seq] = kmat
        return list_of_kmats

    def __len__(self):
        if self.configs['mode']=='train':
            return len(self.train_frames)
        else:
            return len(self.test_frames)

    def __getitem__(self, index):
        if self.configs['mode']=='train':
            src_frame = self.train_frames[index]
        else:
            src_frame = self.test_frames[index]
        src_frame, target_frame = self._get_ref_target_pair(src_frame)
        data_dict = self._read_data(src_frame, target_frame)
        return data_dict

    def _read_data(self, src_frame,  target_frame):
        seq = Path(src_frame).parent.parent.stem
        rot_mat, t_vec = self._read_pose(seq, src_frame,  target_frame)
        src_frame,  target_frame = Path(src_frame), Path(target_frame)
        src_num, target_num = str(src_frame.stem), str(target_frame.stem)
        if self.configs['mode']=='train':
            kmatrix = self.train_kmats[seq].clone()
        else:
            kmatrix = self.test_kmats[seq].clone()
        img_src, _ = self._read_image(src_frame)
        img_target, f_scaling = self._read_image(target_frame)

        kmatrix[0,:] = kmatrix[0,:]*f_scaling[0]
        kmatrix[1,:] = kmatrix[1,:]*f_scaling[1]
        data_dict = {}
        data_dict['input_img'] = img_src
        data_dict['target_img'] = img_target
        data_dict['k_mats'] = kmatrix
        data_dict['r_mats'] = rot_mat
        data_dict['t_vecs'] = t_vec
        return data_dict

    def _read_pose(self, seq, src_frame, target_frame):
        src_frame,  target_frame = Path(src_frame), Path(target_frame)
        seq = src_frame.parent.parent.stem
        src_num, target_num = str(src_frame.stem), str(target_frame.stem)
        if src_num==target_num:
            src_parent, trg_parent = src_frame.parent.stem, target_frame.parent.stem
            r_rel = torch.eye(3)
            t_rel = torch.Tensor([[-0.54], [0.0], [0.0]])
            if src_frame.parent.stem.endswith('image_3'):
                t_rel = torch.Tensor([[0.54], [0.0], [0.0]])
        else:
            if self.configs['mode']=='train':
                seq_pose = self.train_poses[seq]
            else:
                seq_pose = self.test_poses[seq]
            src_pose, target_pose = map(lambda x:torch.from_numpy(x), \
                                        [seq_pose[int(src_num)], seq_pose[int(target_num)]])
            r_s, t_s = src_pose[0:3,0:3], src_pose[0:3, 3]
            r_t, t_t = target_pose[0:3,0:3], target_pose[0:3, 3]
            r_rel = torch.mm(r_t.transpose(1,0), r_s)
            t_rel = torch.mm(r_t.transpose(1,0), (t_s-t_t).view(3,1))
        return r_rel, t_rel

    def _read_image(self, input_path):
        w, h = self.configs['width'], self.configs['height']
        img = cv.imread(str(input_path))
        h_org, w_org, _ = img.shape
        # print(img.shape)
        img = cv.resize(img, (w, h))
        # print(img.shape)
        fx_ratio, fy_ratio = w/w_org, h/h_org
        
        a = - 1
        b = 1
        min_val = 0
        max_val = 255
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        return a + (img - min_val) * (b - a)/(max_val - min_val), [fx_ratio, fy_ratio]

    def _get_ref_target_pair(self, src_frame):
        frame_path = Path(src_frame)
        seq = frame_path.parent.parent.stem
        frame = int(frame_path.stem)
        max_offset = self.configs['max_baseline']
        assert max_offset>0, 'offset should be atleast 1'
        # if True:
        offset = 1
        if(max_offset > 1):
            offset = random.randint(1, max_offset-1)
        if random.random()>0.5:
            offset = -offset
        target_idx = str(frame + offset).zfill(6)
        target_path = frame_path.parent / Path(target_idx+'.png')
        target_frame = str(target_path)
        # else:
        # target_frame = str(frame_path).replace('image_2', 'image_3')
        return src_frame, target_frame


if __name__=='__main__':
    configs = {'mode':'train',
                'max_baseline':10,
                'height':256,
                'width':256,
                }
    loader = KittiLoader(configs)
    print(loader.__len__())
    exit()
    for i in range(0, 20000, 1000):
        data_dict = loader.__getitem__(i)
        for k,v in data_dict.items():
            print(k, v.shape)
        print(data_dict['k_matrix'])
