import os
import os.path as path
import random
from collections import namedtuple

import cv2 as cv
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from . configs import *


def carla_k_matrix(fov=90.0, height=600, width=800):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
                        (2.0 * math.tan(fov * math.pi / 360.0))
    return torch.from_numpy(k)


def resize_segmentation_maps(seg_map, target_size):
    h, w = seg_map.shape
    seg_map = seg_map.view(1, 1, h, w)
    seg_map = F.interpolate(input=seg_map, size=(target_size))
    return seg_map.view(target_size)


def resize_disparity(disp_input, target_size):
    # print(disp_input.min(), disp_input.max())
    # print(disp_input.shape)
    h, w = disp_input.shape
    disp_input = disp_input.view(1, 1, h, w)
    # print('ratio ..... ', target_size[1]/w)
    scaling_factor = target_size[1] / w
    re_scaled_disp = F.interpolate(input=disp_input, size=(target_size),                                mode='bilinear', align_corners=True)
    re_scaled_disp = re_scaled_disp.mul(scaling_factor)
    return re_scaled_disp.view(target_size)


def label_to_one_hot(input_seg, num_classes):
    '''
    pass a in put tensor
    '''
    # num_classes = CARLA_NUM_CLASSES
    assert input_seg.max() < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}'
    b, _, h, w = input_seg.shape
    lables = torch.zeros(b, num_classes, h, w).float()
    labels = lables.scatter_(dim=1, index=input_seg.long(), value=1.0)
    labels = labels.to(input_seg.device)
    return labels


def labels_to_cityscapes_palette(tensor, num_classes=13):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    array = tensor.numpy()
    classes_ = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 255],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]  # TrafficSigns
    }
    classes = {i:classes_[i] for i in range(num_classes)}
    result = torch.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    result = torch.from_numpy(result).permute(2, 0, 1)
    return result


def load_segmentation(input_path):
    img = cv.imread(input_path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # print(input_path)
    try:
        img = np.asarray(img, dtype=np.uint8)
        img = torch.from_numpy(img[..., 2])
    except:
        print(f'cannot read {input_path}')
    return img


def load_depth(input_path):
    # h, w, _ = indices.shape
    img = np.asarray(Image.open(input_path), dtype=np.uint8)
    img = img.astype(np.float64)[:,:,:3]  # .double()
    normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
    # print('depth range ', normalized_depth.min(), normalized_depth.max())
    """
    kinv = torch.inverse(kmat)
    kinv = kinv.view(1, 3, 3).expand(int(h*w), 3, 3)
    depth_vec = normalized_depth.view(int(h*w), 1)
    depth_vec = depth_vec.view(int(h*w), 1, 1)
    depth_vec = depth_vec.expand(int(h*w), 3, 1)
    indices_vectorised = indices.view(int(h*w), 3, 1)
    ray = torch.bmm(kinv, indices_vectorised)
    print(ray[:, 2:, ].max())
    print(ray[:, 2:, ].mean())
    print(ray[:, 2:, ].min())
    ray_norm = torch.norm(ray, dim=1, p=2).view(int(h*w), 1, 1)
    ray_norm = ray_norm.expand(int(h*w), 3, 1)
    ray_unit_len = (ray/ray_norm)
    ray_scaled = ray_unit_len*depth_vec.double()
    depth_z = ray_scaled[:, 2, :]
    #depth_z = ray_norm[:, 2, :]
    #print(depth_z.shape)
    #depth_z = ray*depth_vec.float()
    #depth_z = depth_z[:, 2, 0]
    depth_z = depth_z.view(h, w)
    #print(torch.mean(depth_z.float()-normalized_depth.float()))
    #print(torch.max(depth_z.float()-normalized_depth.float()))
    #exit()
    """
    """
    #print(scaled_ray.)
    #exit()
    return normalized_depth
    return normalized_depth
    """
    return normalized_depth.float()
