import os
import math
import tqdm
import random
import platform
import warnings
from pathlib import Path
import cv2 as cv
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import PIL.Image as Image

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)
from .pfm_rw import read_pfm

from .carla_utils import carla_k_matrix
from .carla_utils import load_segmentation
from .carla_utils import load_depth, resize_disparity
from .carla_utils import load_segmentation, label_to_one_hot
from .carla_utils import resize_segmentation_maps


class CarlaFiveCities(Dataset):
    '''
    carla dataset loader: supports fixed style loading
    '''

    def __init__(self, configs, mode='train'):
        super(CarlaFiveCities, self).__init__()
        configs['mode'] = mode
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.ToPIL = transforms.Compose([transforms.ToPILImage()])
        self.configs = configs

        self.height, self.width = self.configs['height'], self.configs['width']
        self.stereo_baseline = configs['stereo_baseline']
        self.fx_org = self._get_k_matrix()[0, 0]

        self.base_path = '/data/teddy/temporary_carla'
        filtered_test_frames = f'{self.base_path}_test/carla_5_cities_test_split_file.txt'
        if configs['machine_name'] == 'geneva':
            self.base_path = '/netscratch/teddy/temporary_carla'
            filtered_test_frames = f'{self.base_path}_test/carla_5_cities_test_split_file.txt'
        filtered_test_frames = f'{self.base_path}_test/carla_5_cities_test_split_file.txt'
        towns = [f'Town0{x}' for x in range(1, 6)]
        weathers = [f'weather_0{x}' for x in range(4)]
        # camera_groups_prefix = ['SideCameras_', 'HorizontalCameras_']
        camera_groups_prefix = ['ForwardCameras_',
                                'SideCameras_', 'HorizontalCameras_']
        if configs['mode'] == 'train':
            self.cams_per_group = [str(x).zfill(2) for x in range(5)]
        else:
            self.cams_per_group = [str(x).zfill(2) for x in range(5)]
        file_names = [str(x).zfill(6) + '.png' for x in range(0, 10000, 10)]
        examples = []
        if mode == 'train':
            for t in towns:
                for w in weathers:
                    for c in camera_groups_prefix:
                        for f in file_names:
                            examples.append(os.path.join(
                                self.base_path, t, w, c + '00', 'rgb', f))
            self.frames = examples
        if mode == 'test':
            self.frames = []
            with open(filtered_test_frames, 'r') as fid:
                for lines in fid.readlines():
                    lines_ = lines.split(',')
                    lines_ = [n.replace('\n', '') for n in lines_]
                    if not configs['machine_name'].startswith('serv'):
                        lines_ = [
                            l.replace('/data/teddy/temporary', '/netscratch/teddy/temporary') for l in lines_]
                    self.frames.append(lines_[0])
            # if 'include_side_cams' in configs.keys():
            #     if configs['include_side_cams']:
            #         print(len(self.frames))
            #         side_frames = []
            #         for file_ in tqdm.tqdm(self.frames, total=len(self.frames)):
            #             side_cam = file_.replace('ForwardCameras', 'SideCameras')
            #             side_cam = side_cam.replace('HorizontalCameras', 'SideCameras')
            #             side_frames.append(side_cam)
            #         side_frames = list(set(side_frames))
            #         self.frames.extend(side_frames)
            self.frames = sorted(self.frames)

    def __len__(self):
        return len(self.frames)

    def replace_cam_id(self, frame, src_camera):
        frame_parts = frame.split('/')
        frame_parts[6] = frame_parts[6][:-2] + src_camera
        out_path = os.path.join('/', *frame_parts)
        return out_path

    def get_random_frame(self, input_path):
        folder = str(Path(input_path).parent)
        files_ = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files_ if os.path.isfile(f)]
        seed_val = int(input_path.split(
            '/')[4][-2:]) * 10 + int(input_path.split('/')[5][-2:])
        random.seed(seed_val)
        #random_file = files[int(0.35*len(files))]
        random_file = random.choice(files)
        return random_file

    def __getitem__(self, index):
        configs = self.configs
        # if configs['mode']=='train':
        frame = self.frames[index]
        fpath = Path(frame)
        cam_group = fpath.parent.parent.stem
        if configs['mode'] == 'train':
            trg_camera, src_camera = random.sample(self.cams_per_group, 2)
        else:
            trg_camera, src_camera = '01', '00'
        src_frame = self.replace_cam_id(frame, src_camera)
        trg_frame = self.replace_cam_id(frame, trg_camera)
        src_rgb, trg_rgb = self._read_rgb(src_frame), self._read_rgb(trg_frame)
        rot_mat, t_vec = self._get_pose(src_camera, trg_camera, cam_group[:-3])
        kmatrix = self._get_k_matrix(
            height=configs['height'], width=configs['width'])
        old_style_dict = {}
        old_style_dict['t_vecs'] = t_vec.float()
        old_style_dict['k_mats'] = kmatrix.float()
        old_style_dict['r_mats'] = rot_mat.float()
        old_style_dict['input_img'] = src_rgb.float()
        old_style_dict['target_img'] = trg_rgb.float()
        return old_style_dict

    def align_midas_to_mpi(self, inverse_depth, near_plane, far_plane, focal_len, baseline):
        _, h, w = inverse_depth.shape
        inverse_depth = inverse_depth.view(h * w)

        inverse_depth = inverse_depth - inverse_depth.min()
        if inverse_depth.max() < 1e-04:
            inverse_depth = inverse_depth / 1e-04
        else:
            inverse_depth = inverse_depth / inverse_depth.max()
        mpi_inv_depth_min = 1.0 / far_plane
        mpi_inv_depth_max = 1.0 / near_plane
        midas_inv_depth = inverse_depth * \
            (mpi_inv_depth_max - mpi_inv_depth_min) + mpi_inv_depth_min
        midas_inv_depth = midas_inv_depth.view(1, h, w)
        midas_disp = baseline * focal_len * midas_inv_depth
        midas_disp = midas_disp.view(1, h, w)
        # print('midas disp', midas_disp.min(), midas_disp.max())
        return midas_disp

    def get_one_hot_vec(self, input_frame):
        frame_path = Path(input_frame)
        weather = int(frame_path.parent.parent.parent.stem[-2:])
        one_hot = torch.zeros(1, self.configs['one_hot_size']).float()
        one_hot[0, weather] = 1.0
        return one_hot

    def _read_rgb(self, img_path):
        img_org = io.imread(str(img_path))
        if img_org.shape[-1] == 4:
            img_org = img_org[:, :, :-1]
        img_np = cv.resize(img_org, (self.width, self.height)) / 255.0
        if 'zero_to_one' in self.configs.keys():
            if self.configs['zero_to_one']:
                img_np = img_np
            else:
                img_np = (2 * img_np) - 1.0
        else:
            img_np = (2 * img_np) - 1.0
        img_tensor = torch.from_numpy(img_np).transpose(
            2, 1).transpose(1, 0).float()
        return img_tensor

    def _read_seg(self, input_path):
        input_path = input_path.replace('rgb', 'semantic_segmentation')
        seg_org = load_segmentation(input_path).float()
        # seg = resize_segmentation_maps(seg_org, [self.height, self.width]).long()
        # seg = label_to_one_hot(seg.unsqueeze(0).unsqueeze(1), self.configs['num_classes']).squeeze(0)
        seg = resize_segmentation_maps(
            seg_org, [self.height, self.width]).long().unsqueeze(0)
        return seg

    def _read_disp(self, img_file_name):
        disp_file = img_file_name[:-3].replace('rgb', 'midas_depth') + 'pfm'
        if os.path.exists(disp_file):
            disp = read_pfm(disp_file)
            disp = torch.from_numpy(disp.copy()).float().unsqueeze(0)
        else:
            print(f'Not found..... {disp_file}')
            disp = torch.ones(
                1, self.configs['height'], self.configs['width']).float()
        return disp

    # def _read_disp(self, depth_path):
    #     depth_path = depth_path.replace('rgb', 'depth')
    #     depth_img = load_depth(depth_path)
    #     disp_img = self.stereo_baseline * self.fx_org / (depth_img + 0.0000000001)
    #     disp_img = resize_disparity(disp_img, [self.height, self.width])
    #     return disp_img

    def _get_k_matrix(self, height=600, width=800):
        scale_x, scale_y = width / 800.0, height / 600.0
        k_org = self._carla_k_matrix()
        k_org[0, :] = (scale_x) * k_org[0, :]
        k_org[1, :] = (scale_y) * k_org[1, :]
        return k_org

    def _carla_k_matrix(self, fov=90.0, height=600, width=800):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)

    def _get_pose(self, src_cam, trg_cam, cam_group):
        r_mat = torch.eye(3)
        if cam_group == 'HorizontalCameras':
            baseline = self.stereo_baseline * \
                float(int(src_cam[-1]) - int(trg_cam[-1]))
            t_vec = torch.Tensor([[baseline], [0], [0]])
        elif cam_group == 'ForwardCameras':
            baseline = self.stereo_baseline * \
                float(int(src_cam[-1]) - int(trg_cam[-1]))
            t_vec = torch.Tensor([[0], [0], [baseline]])
        elif cam_group == 'SideCameras':
            baseline = self.stereo_baseline * \
                float(int(trg_cam[-1]) - int(src_cam[-1]))
            t_vec = torch.Tensor([[baseline], [0], [0]])
        else:
            raise KeyError(f'No such camera group as {cam_group}')

        return r_mat, t_vec
