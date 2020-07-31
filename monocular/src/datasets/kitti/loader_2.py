import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
import torch.utils.data as data
import os
import csv
import random
from PIL import Image

class KITTIDataLoader(data.Dataset):
    def __init__(self):
        super(KITTIDataLoader, self).__init__()


    def initialize(self, opt):
        self.opt = opt
        self.dataroot = self.opt['dataset_root']

        self.bound = self.opt['max_baseline']

        if(self.opt['mode'] == 'train'):
            with open(os.path.join(self.dataroot, 'id_train.txt'), 'r') as fp:
                self.ids_train = [s.strip() for s in fp.readlines() if s]

            self.ids = self.ids_train
        else:
            with open(os.path.join(self.dataroot, 'id_test.txt'), 'r') as fp:
                self.ids_test = [s.strip() for s in fp.readlines() if s]

            self.ids = self.ids_test

        self.dataset_size = int(len(self.ids))// (self.bound*2)

        self.pose_dict = {}
        pose_path = os.path.join(self.dataroot, 'poses.txt')
        with open(pose_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                id = row[0]
                self.pose_dict[id] = []
                for col in row[1:-1]:
                    self.pose_dict[id].append(float(col))
                self.pose_dict[id] = np.array(self.pose_dict[id])

    def __getitem__(self, index):
        id = self.ids[index]
        id_num = int(id.split('_')[-1])
        while True:
            delta = random.choice([x for x in range(-self.bound, self.bound+1) if x != 0] )
            id_target = id.split('_')[0] +'_' + str(id_num + delta).zfill(len(id.split('_')[-1]))
            if id_target in self.pose_dict.keys(): break

        B = self.load_image(id) / 255. * 2 - 1
        B = torch.from_numpy(B.astype(np.float32)).permute((2,0,1))
        A = self.load_image(id_target) / 255. * 2 - 1
        A = torch.from_numpy(A.astype(np.float32)).permute((2,0,1))

        poseB = self.pose_dict[id]
        poseA = self.pose_dict[id_target]
        TB = poseB[3:].reshape(3, 1)
        RB = ROT.from_euler('xyz',poseB[0:3]).as_dcm()
        TA = poseA[3:].reshape(3, 1)
        RA = ROT.from_euler('xyz',poseA[0:3]).as_dcm()
        T = RA.T.dot(TB-TA)/50.

        # Changing the rotation and translation to our convention
        R = RA.T@RB
        R = R.T
        T = R.dot(-T)

        # mat = np.block(
        #     [ [RA.T@RB, T],
        #       [np.zeros((1,3)), 1] ] )

        data_dict = {}
        data_dict['input_img'] = torch.Tensor(A)
        data_dict['target_img'] = torch.Tensor(B)
        data_dict['k_mats'] = torch.Tensor([718.9, 0., 128, 0., 718.9, 128, 0., 0., 1.]).reshape((3, 3))
        data_dict['r_mats'] = torch.Tensor(R)
        data_dict['t_vecs'] = torch.Tensor(T)

        return data_dict
        # return {'A': A, 'B': B, 'RT': mat.astype(np.float32)}

    def load_image(self, id):
        image_path = os.path.join(self.dataroot, 'images', id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        return image

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'KITTIDataLoader'