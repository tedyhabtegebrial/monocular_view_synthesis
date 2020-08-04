"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# class AlphaComposition(nn.Module):
#     '''This class implements alpha compostion.
#     Accepts input warped images and their alpha channels. We perform A-over-B composition in back-to-front manner.......
#     :param src_imgs: tensor of shape [B, D, 3, H, W] where B, D, and [H,W] are batch size, number of mpi planes, and image dimensions, respectively
#     :param alpha_imgs: tensor of shape [B, D, 1, H, W] where B, D, and [H,W] are batch size, number of mpi planes, and image dimensions, respectively
#     :return: comp_rgb result of alpha composition, has shape [B, 3, H, W]
#     '''
#
#     def __init__(self):
#         super(AlphaComposition, self).__init__()
#
#     def forward(self, src_imgs, alpha_imgs):
#         b_size, num_d, _c, h, w = src_imgs.shape
#         src_imgs = torch.split(src_imgs, split_size_or_sections=1, dim=1)
#         alpha_imgs = torch.split(alpha_imgs, split_size_or_sections=1, dim=1)
#         comp_rgb = src_imgs[-1] * alpha_imgs[-1]
#         for d in reversed(range(num_d - 1)):
#             comp_rgb = src_imgs[d] * alpha_imgs[d] + (1.0 - alpha_imgs[d]) * comp_rgb
#         return comp_rgb.squeeze(1)

class AlphaComposition(nn.Module):
    def __init__(self, configs=None):
        super(AlphaComposition, self).__init__()

    def compute_visibility(self, alpha):
        alpha = torch.split(alpha, split_size_or_sections=1, dim=1)
        resistance = [torch.ones_like(alpha[0])]
        visibility = [alpha[0]]
        for i in range(1, len(alpha)):
            res = resistance[-1] * (1.0 - alpha[i - 1])
            resistance.append(res)
            visibility.append(res * alpha[i])
        visibility = torch.cat(visibility, dim=1)
        vis_sum = visibility.sum(dim=1, keepdim=True).clamp(min=0.0000001)
        visibility = visibility / vis_sum
        return visibility

    def forward(self, src_imgs, alpha):
        visibility = self.compute_visibility(alpha)
        output = torch.sum(src_imgs * visibility, dim=1, keepdim=False)
        return output
