import torch

class ComputeBlendingWeights():
	def __call__(self, alphas):
		device = alphas.device
		b, d, _, h, w = alphas.shape
		blending_weights = [torch.ones((b, 1, 1, h, w)).to(device)]
		alphas = torch.split(alphas, dim=1, split_size_or_sections=1)
		# print(alphas[:, 2-1, :, :].unsqueeze(1).shape)
		for d in range(1, len(alphas)):
			blend_w = blending_weights[-1]*(1.0 - alphas[d-1])
			blending_weights.append(blend_w)
			# blending_weights = torch.cat([blending_weights, blending_weights[:,-1, :, :].unsqueeze(1) * alphas[:, d-1, :, :].unsqueeze(1)], dim=1)
			# print(blending_weights.shape)
		blending_weights = torch.cat(blending_weights, dim=1)
		return blending_weights


if __name__ == '__main__':
	comp = ComputeBlendingWeights()
	alphas = torch.rand((4, 32, 1, 200, 200))
	blending_weights = comp(alphas)
	print(blending_weights.shape)