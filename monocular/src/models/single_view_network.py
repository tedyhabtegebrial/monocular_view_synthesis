import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewNetwork(nn.Module):

	def __init__(self, configs):
		super(SingleViewNetwork, self).__init__()

		self.conv_1_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
		self.conv_1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, padding=3)

		self.conv_2_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
		self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)

		self.conv_3_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=118, kernel_size=3, padding=1)

		self.conv_4_0 = nn.Conv2d(in_channels=118, out_channels=256, kernel_size=3, padding=1)
		self.conv_4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

		self.conv_5_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_6_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv_6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_7_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv_7_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_8_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv_8_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_9_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
		self.conv_9_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_10_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
		self.conv_10_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_11_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
		self.conv_11_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_12_0 = nn.Conv2d(in_channels=512 + 256, out_channels=512, kernel_size=3, padding=1)
		self.conv_12_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

		self.conv_13_0 = nn.Conv2d(in_channels=512 + 118, out_channels=128, kernel_size=3, padding=1)
		self.conv_13_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

		self.conv_14_0 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1)
		self.conv_14_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

		self.conv_15_0 = nn.Conv2d(in_channels=32 + 64, out_channels=64, kernel_size=3, padding=1)
		self.conv_15_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

		self.conv_16_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.conv_16_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.batch16 = nn.BatchNorm2d(64)
		self.output = nn.Conv2d(in_channels=64, out_channels=configs['num_planes'] + 2, kernel_size=3, padding=1)

	def forward(self, rgb):
		# Encoding
		conv1 = F.relu(self.conv_1_1(F.relu(self.conv_1_0(rgb))))

		conv2 = F.relu(self.conv_2_1(F.relu(self.conv_2_0(F.max_pool2d(conv1, kernel_size=2)))))

		conv3 = F.relu(self.conv_3_1(F.relu(self.conv_3_0(F.max_pool2d(conv2, kernel_size=2)))))

		conv4 = F.relu(self.conv_4_1(F.relu(self.conv_4_0(F.max_pool2d(conv3, kernel_size=2)))))

		conv5 = F.relu(self.conv_5_1(F.relu(self.conv_5_0(F.max_pool2d(conv4, kernel_size=2)))))

		conv6 = F.relu(self.conv_6_1(F.relu(self.conv_6_0(F.max_pool2d(conv5, kernel_size=2)))))

		conv7 = F.relu(self.conv_7_1(F.relu(self.conv_7_0(F.max_pool2d(conv6, kernel_size=2)))))

		conv8 = F.relu(self.conv_8_1(F.relu(self.conv_8_0(F.max_pool2d(conv7, kernel_size=2)))))

		# Decoding
		up_8 = F.interpolate(conv8, scale_factor=2, mode='nearest')
		conv9 = F.relu(self.conv_9_1(F.relu(self.conv_9_0(torch.cat([up_8, conv7], dim=1)))))

		up_9 = F.interpolate(conv9, scale_factor=2, mode='nearest')
		conv10 = F.relu(self.conv_10_1(F.relu(self.conv_10_0(torch.cat([up_9, conv6], dim=1)))))

		up_10 = F.interpolate(conv10, scale_factor=2, mode='nearest')
		conv11 = F.relu(self.conv_11_1(F.relu(self.conv_11_0(torch.cat([up_10, conv5], dim=1)))))

		up_11 = F.interpolate(conv11, scale_factor=2, mode='nearest')
		conv12 = F.relu(self.conv_12_1(F.relu(self.conv_12_0(torch.cat([up_11, conv4], dim=1)))))

		up_12 = F.interpolate(conv12, scale_factor=2, mode='nearest')
		conv13 = F.relu(self.conv_13_1(F.relu(self.conv_13_0(torch.cat([up_12, conv3], dim=1)))))

		up_13 = F.interpolate(conv13, scale_factor=2, mode='nearest')
		conv14 = F.relu(self.conv_14_1(F.relu(self.conv_14_0(torch.cat([up_13, conv2], dim=1)))))

		up_14 = F.interpolate(conv14, scale_factor=2, mode='nearest')
		conv15 = F.relu(self.conv_15_1(F.relu(self.conv_15_0(torch.cat([up_14, conv1], dim=1)))))

		conv16 = F.relu(self.conv_16_1(F.relu(self.conv_16_0(conv15))))
		conv16 = self.batch16(conv16)
		
		output = self.output(conv16)
		print('alpha values:', output[:, :-3, :, :].min().item(), output[:, :-3, :, :].max().item())

		return F.sigmoid(output)




