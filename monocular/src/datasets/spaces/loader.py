import os
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
from utils import ReadScene as read_scene

class SpacesLoader(Dataset):
    def __init__(self, mode, data_path):
        super(self, SpacesLoader).__init__()
        self.data_path = data_path
        folders_list = []
        if mode=='train':
            folders_list.extend(self.get_folders('800'))
            folders_list.extend(self.get_folders('2k'))
        else:
            folders_list.extend(self.get_folders('eval'))
        self.folders_list = folders_list
    def get_folders(self, folder_path):
        data_path = self.data_path
        files_1 = os.listdir(os.path.join(data_path, folder_path))
        files_1 = [os.path.join(data_path, folder_path, f) for f in files_1]
        files_1 = [f for f in files_1 if os.isdir(f)]
        return files_1

    def __len__(self):
        return len(self.folders_list)

    def __getitem__(self,index):
        current_folder = self.folders_list[index]
        scene_views = read_scene(current_folder)
        # each scene contains the following
        for view in scene_views:
            print(view)


if __name__=='__main__':
    loader = SpacesLoader('/home/habtegebrial/Desktop/Work/repos/spaces_dataset-master/data')
    loader.__getitem__(10)
