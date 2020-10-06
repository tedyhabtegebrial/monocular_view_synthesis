import tqdm
import os, math
import random, platform
from pathlib import Path
import cv2 as cv
from pathlib import Path
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import PIL.Image as Image

from carla_utils import load_depth

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)


test_folder = '/data/teddy/temporary_carla_test'
test_examples = []

towns = [f'Town0{x}' for x in range(1, 5)]
weathers = [f'weather_0{x}' for x in range(4)]

file_names = [str(x).zfill(6)+'.png' for x in range(0, 10000, 10)]
for t,w in zip(towns, weathers):
    for f in file_names:
        c = random.choice(['ForwardCameras_', 'HorizontalCameras_'])
        test_examples.append(os.path.join(test_folder, t, w, c+'00', 'depth', f))
        test_examples.append(os.path.join(test_folder, t, w, 'SideCameras_00', 'depth', f))
        
filtered_examples = []
print(f'Total Number of files === {len(test_examples)}')
for file in tqdm.tqdm(test_examples):
    depth = load_depth(file)
    if depth.min()>1.25:
        filtered_examples.append(file)
        

def find_target(input_file):
    file_parts = input_file.split('/')
    x = random.choice([1,2,3,4])
    target_cam = file_parts[-3][:-2]+str(x).zfill(2)
    target_file = input_file.replace(file_parts[-3], target_cam)
    swap_possible = not file_parts[-3].startswith('ForwardCameras_')
    return target_file, swap_possible

# Now we should select a pair as output
src_target_pairs = []
for file in filtered_examples:
    src_file = file.replace('depth', 'rgb')
    target_file, swap = find_target(src_file)
    if swap and np.random.rand()>0.5:
        src_file, target_file = target_file, src_file
    src_target_pairs.append([src_file, target_file])

with open(f'{test_folder}/carla_5_cities_test_split_file_version_2.txt', 'w') as fid:
    for pairs in src_target_pairs:
        line = pairs[0] + ',' + pairs[1] + '\n'
        fid.writelines(line)
