"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import torch
import time
import torch.nn as nn

from .mpi import ComputeHomography
from .mpi import ApplyHomography
from .mpi import AlphaComposition
from .mpi import ComputeBlendingWeights
from .mpi import WarpWithFlowFields
from .models import BackgroundNetwork, SingleViewNetwork_DFKI, ConvNetwork, SingleViewNetwork
import torchvision


class StereoMagnification(nn.Module):
    '''This is a class that can perform monocular view synthesis. Given a reference color image, camera intrinsics and extrinsics it returns a novel view.
    :param configs: a python dictionary that contains necessary configurations
    '''

    def __init__(self, configs):
        '''Constructor method
        '''
        super(StereoMagnification, self).__init__()
        self.configs = configs

        self.mpi_net = SingleViewNetwork(configs)
        self.background = BackgroundNetwork(configs)
        self.reduce_high_features = BackgroundNetwork(configs, reduce=True)

        self.compute_blending_weights = ComputeBlendingWeights()
        self.compute_homography = ComputeHomography(configs)
        self.apply_homography = ApplyHomography()
        self.composite = AlphaComposition()
        self.warp_with_ff = WarpWithFlowFields(configs)

    def _get_color_imgs_per_plane(self, fg, bg, weight):
        b, z, _, h, w = weight.shape
        fg = fg.unsqueeze(1)
        bg = bg.unsqueeze(1)
        # color_imgs = bg*weight + (1.0-weight)*fg
        color_imgs = fg * weight + (1.0 - weight) * bg
        return color_imgs

# Forward function for using single view DFKI
    def forward(self, input_img, kmats, rmats, tvecs):
        # print(input_img.shape)
        b, c, h, w = input_img.shape
        # torch.cuda.synchronize()
        #t_start = time.time()
        # for param in self.mpi_net.parameters():
        #     print(param.data)
        alphas_assoc = self.mpi_net(input_img)
        # torch.cuda.synchronize()
        #print('compute alphas:', time.time() - t_start)
        alphas = alphas_assoc[:,
                              :self.configs['num_planes'], :, :].unsqueeze(2)
        assoc = alphas_assoc[:, self.configs['num_planes']:, :, :].view(
            -1, self.configs['num_planes'], self.configs['occlusion_levels'], h, w)
        # print('assoc shape', assoc.shape)
        # torch.cuda.synchronize()
        # t_start = time.time()
        mult_layer_features = self.background(input_img)
        # torch.cuda.synchronize()
        # print('compute feats:', time.time() - t_start)
        # torch.cuda.synchronize()
        # t_start = time.time()
        h_mats = self.compute_homography(kmats, rmats, tvecs)
        # print('hmats', h_mats.min().item(), h_mats.max().item())
        # torch.cuda.synchronize()
        # print('compute hom:', time.time() - t_start)
        # t_start = time.time()
        warped_features, alphas = self._get_warped_features(
            h_mats, alphas, assoc, mult_layer_features)
        # print('warped_features', warped_features.min().item(), warped_features.max().item())
        # print('warped_alphas', alphas.min().item(), alphas.max().item())
        # torch.cuda.synchronize()
        # print('Warp feats:', time.time() - t_start)
        # t_start = time.time()
        rgb_img = self.reduce_high_features(warped_features)
        # print('rgb', rgb_img.min().item(), rgb_img.max().item())
        # torch.cuda.synchronize()
        # print('feat 2 rgb hom:', time.time() - t_start)
        return rgb_img, alphas


# # Forward function for using single view
#     def forward(self, input_img, kmats, rmats, tvecs):
#         '''The pytorch interface __call__ function. All inputs are tensors
#     #     :param input_img: \\in [B, 3, H, W] where B is batch size, H,W are image dimensions
#     #     :param kmats:     Camera intriciscs  \\in [B, 3, 3]
#     #     :param rmats:     Camera intriciscs  \\in [B, 3, 3] Rotation matrix from source camera to the target camera
#     #     :param tvecs:     Translation vector  \\in [B, 3, 1] When the source camera center is expressed in the target camera coordinate frame
#     #     :return pred_img: novel view color image
#     #     :return alphas:   per-plane alphas, returned so that we can see what the scene representation looks like
#         '''
#         # output of net is [B,H,W,C]
#         mpi_alphas_bg_img = self.mpi_net(input_img)
#         # b, h, w, c = mpi_alphas_bg_img.shape
#         alpha = mpi_alphas_bg_img.permute(
#             [0, 3, 1, 2])[:, :-3, ...].unsqueeze(-1)
#         layer_alpha = torch.cat(
#             [torch.ones_like(alpha[:, 0:1]), alpha], axis=1)
#         # b, c, h, w = input_img.shape
#         fg_img = input_img.unsqueeze(1)
#         bg_img = mpi_alphas_bg_img[:, :, :, -
#                                    3:].permute([0, 3, 1, 2]).unsqueeze(1)
#         # create blending weight from alpha values by exclusive reversed cumulative product
#         # blending weights shape is [B, D, 1, H, W]
#         blending_weights = self.compute_blending_weights(layer_alpha)
#         # print('blending_weights', blending_weights.shape)
#         # print('fg_img', fg_img.shape)
#         # print('bg_img', bg_img.shape)
#         # flipped_alphas = torch.flip(layer_alpha, dims=[1])
#         # ones_ = torch.ones_like(flipped_alphas)[:,:1,...]
#         # blending_weights = torch.cumprod(1.0 - flipped_alphas, axis = 1)[:,:-1,...]
#         # blending_weights = torch.cat([ones_, blending_weights], axis=1)
#         # blending_weights = torch.flip(blending_weights, dims=[1])
#
#         layer_rgb = blending_weights * fg_img + \
#             (1.0 - blending_weights) * bg_img
#         # layers = torch.cat([layer_rgb, layer_alpha], axis = -1)
#         h_mats = self.compute_homography(kmats, rmats, tvecs)
#
#         # b, l, h, w, c = layer_alpha.shape
#         layer_alpha = layer_alpha.permute([0, 1, 4, 2, 3])
#         # b, l, h, w, c = layer_rgb.shape
#         # layer_rgb = layer_rgb.permute([0,1,4,2,3])
#
#         #print('layer alpha', layer_alpha.shape)
#         #print('layer rgb', layer_rgb.shape)
#         rgb_img, alphas = self._render_rgb(h_mats, layer_alpha, layer_rgb)
#
#         if self.training:
#             return rgb_img, alphas
#         else:
#             return rgb_img
#          #b, d, h, w = mpi_alphas_bg_img.shape
#          # print('mpi_alphas_bg_img', mpi_alphas_bg_img.shape)
#          #ones_ = torch.ones(b, 1, 1, h, w).to(mpi_alphas_bg_img.device)
#
#          #mpi_alpha = mpi_alphas_bg_img[:, :self.configs['num_planes']-1, :, :].unsqueeze(2)
#          #mpi_alpha = torch.cat([mpi_alpha, ones_], dim=1)
#
#          #blending_weights = mpi_alphas_bg_img[:, self.configs['num_planes']-1:-3, :, :].unsqueeze(2)
#          #blending_weights = torch.cat([ones_.clone(), blending_weights], dim=1)
#
#          #bg_img = mpi_alphas_bg_img[:, -3:, :, :]
#          # print('mpi_alpha', mpi_alpha.shape)
#          # print('blending_weights', blending_weights.shape)
#          # print('bg_img', bg_img.shape)
#          #blending_weights = self.ComputeBlendingWeights(mpi_alpha)
#
#     #      mpi_alpha, blending_weights, bg_img = self.mpi_net(input_img)
#     #      h_mats = self.compute_homography(kmats, rmats, tvecs)
#     #      fg_img = input_img
#     #      # here mistake
#     #      color_imgs_ref_cam = self._get_color_imgs_per_plane(fg_img, bg_img, blending_weights)
#     #      pred_img, alphas = self._render_rgb(h_mats, mpi_alpha, color_imgs_ref_cam)
#     # #     # print(tvecs)
#     #
#     #      if self.training:
#     #          return pred_img, alphas
#     #      else:
#     #          return pred_img

    def _render_rgb(self, h_mats, mpi_alpha_seg, color_imgs_ref_cam):
        # alphas = torch.sigmoid(mpi_alpha_seg)
        alphas = mpi_alpha_seg
        color_imgs = self.apply_homography(h_mats, color_imgs_ref_cam)
        warped_alphas = self.apply_homography(h_mats, alphas)
        output_rgb = self.composite(color_imgs, warped_alphas)
        return output_rgb, alphas

    def _get_warped_features(self, h_mats, alphas, associations, mult_layer_features):
        b, d, _, h, w = alphas.shape
        l = associations.shape[2]
        warped_alphas = self.apply_homography(h_mats, alphas.contiguous())
        warped_assoc = self.apply_homography(
            h_mats, associations, grid=self.apply_homography.grid)
        warped_mult_layer_features = self.warp_with_ff(
            h_mats, mult_layer_features, warped_alphas).reshape(b, l, self.configs['num_features'], h, w)
        composite_assoc = self.composite(warped_assoc, warped_alphas)
        composite_assoc = composite_assoc / \
            torch.sum(composite_assoc, dim=1, keepdim=True).clamp(min=1e-06)
        warped_features = warped_mult_layer_features * \
            composite_assoc.unsqueeze(2)
        return warped_features.sum(dim=1, keepdim=False), alphas


if __name__ == '__main__':
    configs = {}
    configs['width'] = 256
    configs['height'] = 256
    configs['batch_size'] = 1
    configs['num_planes'] = 64
    configs['occlusion_levels'] = 3
    configs['num_features'] = 16
    configs['near_plane'] = 5
    configs['far_plane'] = 10000
    configs['encoder_features'] = 32
    configs['encoder_ouput_features'] = 64
    configs['input_channels'] = 3
    configs['out_put_channels'] = 3
    monocular_nvs_network = StereoMagnification(configs).eval()
    input_img = torch.rand(1, 3, 256, 256)
    k_mats = torch.rand(1, 3, 3)
    r_mats = torch.rand(1, 3, 3)
    t_vecs = torch.rand(1, 3, 1)
    novel_view = monocular_nvs_network(input_img, k_mats, r_mats, t_vecs)
    print(f'Novel VIew Shape== {novel_view.shape}')
