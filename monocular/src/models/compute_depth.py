import torch
import torch.nn as nn


class ComputeDepth(nn.Module):

	def __init__(self, configs):
		super(ComputeDepth, self).__init__()

		self.configs = configs
		self.occlusion_points = torch.Tensor(configs['occlusion_points'])


	def forward(self, image, kmats, hom_mat, r_mats, t_vecs):
		h, w = self.configs['height'], self.configs['width']
		b = image.shape[0]
		device = image.device
		self.occlusion_points.to(device)

		# kmats shape = [B, 3, 3]
		kinv = torch.stack([torch.inverse(k) for k in kmats])
		kinv = kinv.unsqueeze(1)
		kinv = kinv.expand(-1, h * w, 3, 3)
		kinv = kinv.reshape((b * h * w, 3, 3))
		print('K_inv shape', kinv.shape)

		# image shape [B, H, W, 3]
		image = image.reshape((b * h * w, 3, 1))
		print('Image shape', image.shape)
		
		# k_inv_img shape should be [B x H x W, 3, 1]
		k_inv_img = torch.bmm(kinv, image)
		print('K_inv_img Shape', k_inv_img.shape)


		# occlusion points shape = [B x H x W, 1, occlusion levels]
		self.occlusion_points = self.occlusion_points.unsqueeze(0).unsqueeze(0)
		self.occlusion_points = self.occlusion_points.expand(k_inv_img.shape[0], -1, -1)
		print('Occlusion Points shape', self.occlusion_points.shape)
		
		# image in 3d shape [B x H x W, 3, occlusion levels]
		image_3d = torch.bmm(k_inv_img, self.occlusion_points)
		print('Image 3d shape', image_3d.shape)

		# homography matrix
		hom_mat = hom_mat.unsqueeze(0)
		hom_mat = hom_mat.expand(image_3d.shape[0], -1, -1)
		print('Homography matrix shape', hom_mat.shape)

		warped_img = torch.bmm(hom_mat, image_3d)
		print('Warped image shape', warped_img.shape)

		# rt = torch.zeros(b, 3, 4).to(device)
		# rt[:, :3, :3] = r_mats
		# rt[:, :3, 3:] = t_vecs

		r = r_mats.unsqueeze(1)
		r = r.expand(-1, h * w, -1, -1)
		r = r.reshape((image_3d.shape[0], 3, 3))
		print('rotations shape', r.shape)


		# print('Transpose image 3d shape', torch.transpose(image_3d, 1, 2).shape)
		# Image in target coordinates
		image_target = torch.bmm(r, image_3d)
		print('Image Target rotated shape', image_target.shape)

		image_target = image_target.reshape((b, configs['num_occlusion_points'], 3, h, w))
		print('Final target image shape', image_target.shape)

if __name__ == '__main__':
	configs = {}
	configs['occlusion_points'] = [1, 3, 4, 5]
	configs['num_occlusion_points'] = len(configs['occlusion_points'])
	configs['height'] = 10
	configs['width'] = 10

	images = torch.rand(4, 10, 10, 3)
	kmats = torch.rand(4, 3, 3)
	hom_mat = torch.rand(3, 3)
	r_mats = torch.rand(4, 3, 3)
	t_vecs = torch.rand(4, 3, 1)
	network = ComputeDepth(configs).eval()

	network(images, kmats, hom_mat, r_mats, t_vecs)



