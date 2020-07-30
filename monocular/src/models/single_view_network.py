import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleViewNetwork(nn.Module):

	def __init__(self):
        super(SingleViewNetwork, self).__init__()

		self.conv_1_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)
		self.conv_1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7)

		self.conv_2_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
		self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)

		self.conv_3_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=118, kernel_size=3)

		self.conv_4_0 = nn.Conv2d(in_channels=118, out_channels=256, kernel_size=3)
		self.conv_4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

		self.conv_5_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
		self.conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_6_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
		self.conv_6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_7_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
		self.conv_7_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_8_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
		self.conv_8_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_9_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
		self.conv_9_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_10_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
		self.conv_10_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_11_0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
		self.conv_11_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_12_0 = nn.Conv2d(in_channels=512 + 256, out_channels=512, kernel_size=3)
		self.conv_12_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

		self.conv_13_0 = nn.Conv2d(in_channels=512 + 118, out_channels=128, kernel_size=3)
		self.conv_13_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

		self.conv_14_0 = nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3)
		self.conv_14_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

		self.conv_15_0 = nn.Conv2d(in_channels=32 + 64, out_channels=64, kernel_size=3)
		self.conv_15_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

		self.conv_16_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
		self.conv_16_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

		self.output = nn.Conv2d(in_channels=64, out_channels=34, kernel_size=3)

	def forward(self, rgb):
		conv1 = F.max_pool2d(F.relu(self.conv_1_1(F.relu(self.conv_1_0(rgb)))), kernel_size=2)

		conv2 = F.max_pool2d(F.relu(self.conv_2_1(F.relu(self.conv_2_0(conv1)))), kernel_size=2)

		conv3 = F.max_pool2d(F.relu(self.conv_3_1(F.relu(self.conv_3_0(conv2)))), kernel_size=2)

		conv4 = F.max_pool2d(F.relu(self.conv_4_1(F.relu(self.conv_4_0(conv3)))), kernel_size=2)

		conv5 = F.max_pool2d(F.relu(self.conv_5_1(F.relu(self.conv_5_0(conv4)))), kernel_size=2)

		conv6 = F.max_pool2d(F.relu(self.conv_6_1(F.relu(self.conv_6_0(conv5)))), kernel_size=2)

		conv7 = F.max_pool2d(F.relu(self.conv_7_1(F.relu(self.conv_7_0(conv6)))), kernel_size=2)

		conv8 = F.max_pool2d(F.relu(self.conv_8_1(F.relu(self.conv_8_0(conv7)))), kernel_size=2)

		up_8 = F.interpolate(conv8, scale_factor=2, mode='nearest')
		conv9 = F.max_pool2d(F.relu(self.conv_9_1(F.relu(self.conv_9_0(torch.cat([up_8, conv7], dim=1))))), kernel_size=2)

		up_9 = F.interpolate(conv9, scale_factor=2, mode='nearest')
		conv10 = F.max_pool2d(F.relu(self.conv_10_1(F.relu(self.conv_10_0(torch.cat([up_9, conv6], dim=1))))), kernel_size=2)

		up_10 = F.interpolate(conv10, scale_factor=2, mode='nearest')
		conv11 = F.max_pool2d(F.relu(self.conv_11_1(F.relu(self.conv_11_0(torch.cat([up_10, conv5], dim=1))))), kernel_size=2)

		up_11 = F.interpolate(conv11, scale_factor=2, mode='nearest')
		conv12 = F.max_pool2d(F.relu(self.conv_12_1(F.relu(self.conv_12_0(torch.cat([up_11, conv4], dim=1))))), kernel_size=2)

		up_12 = F.interpolate(conv12, scale_factor=2, mode='nearest')
		conv13 = F.max_pool2d(F.relu(self.conv_13_1(F.relu(self.conv_13_0(torch.cat([up_12, conv3], dim=1))))), kernel_size=2)

		up_13 = F.interpolate(conv13, scale_factor=2, mode='nearest')
		conv14 = F.max_pool2d(F.relu(self.conv_14_1(F.relu(self.conv_14_0(torch.cat([up_13, conv2], dim=1))))), kernel_size=2)

		up_14 = F.interpolate(conv14, scale_factor=2, mode='nearest')
		conv15 = F.max_pool2d(F.relu(self.conv_15_1(F.relu(self.conv_15_0(torch.cat([up_14, conv1], dim=1))))), kernel_size=2)

		conv16 = F.relu(self.conv_16_1(F.relu(self.conv_16_0(conv15))))

		output = F.sigmoid(self.output(conv16))

		return output




