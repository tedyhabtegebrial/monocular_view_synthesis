import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgroundNetwork(nn.Module):
	def __init__(self, configs, reduce=False):
		super(BackgroundNetwork, self).__init__()

		self.configs = configs
		self.reduce = reduce

		input_size = 3
		output_size = self.configs['num_features'] * self.configs['occlusion_levels']

		if(reduce):
			input_size = self.configs['num_features']
			output_size = 3

		self.conv_1_0 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=7, padding=3)
		self.conv_1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, padding=3)
		self.bn_1 = nn.BatchNorm2d(32)

		self.conv_2_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
		self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
		self.bn_2 = nn.BatchNorm2d(64)

		self.conv_3_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.bn_3 = nn.BatchNorm2d(118)

		self.conv_4_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.conv_4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.bn_4 = nn.BatchNorm2d(256)

		self.conv_5_0 = nn.Conv2d(in_channels=256 + 128, out_channels=128, kernel_size=3, padding=1)
		self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.bn_5 = nn.BatchNorm2d(128)

		self.conv_6_0 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1)
		self.conv_6_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.bn_6 = nn.BatchNorm2d(64)

		self.conv_7_0 = nn.Conv2d(in_channels=64 + 32, out_channels=output_size, kernel_size=3, padding=1)
		self.conv_7_1 = nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=3, padding=1)
		self.bn_7 = nn.BatchNorm2d(output_size)

		self.output = nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=3, padding=1)

	def forward(self, rgb):
		# Encoding
		conv1 = self.bn_1(F.relu((self.conv_1_1(F.relu(self.conv_1_0(rgb))))))

		conv2 = self.bn_2(F.relu((self.conv_2_1(F.relu(self.conv_2_0(F.max_pool2d(conv1, kernel_size=2)))))))

		conv3 = self.bn_3(F.relu((self.conv_3_1(F.relu(self.conv_3_0(F.max_pool2d(conv2, kernel_size=2)))))))

		conv4 = self.bn_4(F.relu((self.conv_4_1(F.relu(self.conv_4_0(F.max_pool2d(conv3, kernel_size=2)))))))

		# Decoding
		up_4 = F.interpolate(conv4, scale_factor=2, mode='nearest')
		conv5 = self.bn_5(F.relu((self.conv_5_1(F.relu(self.conv_5_0(torch.cat([up_4, conv3], dim=1)))))))

		up_5 = F.interpolate(conv5, scale_factor=2, mode='nearest')
		conv6 = self.bn_6(F.relu((self.conv_6_1(F.relu(self.conv_6_0(torch.cat([up_5, conv2], dim=1)))))))

		up_6 = F.interpolate(conv6, scale_factor=2, mode='nearest')
		conv7 = self.bn_7(F.relu((self.conv_7_1(F.relu(self.conv_7_0(torch.cat([up_6, conv1], dim=1)))))))

		output = self.output(conv7)

		if(not self.reduce):
			return torch.sigmoid(output)

		return torch.tanh(output)


if __name__ == '__main__':
	configs = {}
	configs['width'] = 256
	configs['height'] = 256
	configs['num_features'] = 16
	configs['input_channels'] = 3
	configs['occlusion_levels'] = 3
	network = BackgroundNetwork(configs).eval()
	input_img = torch.rand(1, 3, 256, 256)
	high_dimension_imgs = network(input_img)
	print(f'New Image Shape == {high_dimension_imgs.shape}')
