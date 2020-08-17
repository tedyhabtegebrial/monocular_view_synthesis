from torch.utils.data import Dataset
import numpy as np
import os, glob

class RealEstateLoader(Dataset):
	def __init__(self, configs):
		self.configs = configs
		self.dir = self.configs['dataset_root'] #/home5/anwar/data/realestate10k/
		self.mode = configs['mode']
		self.indices = np.loadtxt(os.path.join(self.dir, 'valid_folders_%s.txt' %self.mode) , dtype=np.str)
		self.indices = self.indices.reshape((-1,))
		self.frames = sorted([glob.glob(os.path.join(self.dir, 'extracted', self.mode, id, '*.jpg'))] for id in self.indices)
		self.text = sorted([glob.glob(os.path.join(self.dir, 'text_files', self.mode, id, '*.txt'))] for id in self.indices)
		# self.frames = [os.path.join(self.dir, 'extracted', self.mode, clip, frame) for clip in self.indices] 
		# self.text = [os.path.join(self.dir, 'text_files', self.mode, clip, frame) for clip in self.indices]

		self.min_angle = 5
		self.min_trans = 0.15

	def __getitem__(self, index):
		return 0

	def __len__(self):
		return len(self.indices)



if __name__ == '__main__':
	configs = {}
	configs['dataset_root'] = '/home/anwar/Desktop/realestate10k'
	configs['mode'] = 'train'
	
	dataset = RealEstateLoader(configs)
	print(len(dataset))
