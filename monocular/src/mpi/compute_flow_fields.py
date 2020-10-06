import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .apply_homography import ApplyHomography
from .alpha_composition import AlphaComposition


class ComputeFlowFields(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.apply_h_mats = ApplyHomography()
        self.alpha_composite = AlphaComposition()
        b_size, num_d = configs['batch_size'], configs['num_planes']
        h, w = configs['height'], configs['width']
        self.xy_locations = self.apply_h_mats.get_homogeneous_xy_locations(
            b_size * num_d, h, w).to('cuda:0')
        self.xy_locations = self.xy_locations.view(
            b_size * num_d, h, w, 3, 1).view(-1, 3, 1)

    def get_grid(self, h_matrix, h, w):
        device_ = h_matrix.device
        b_size = h_matrix.shape[0]
        num_d = h_matrix.shape[1]
        h_matrix = h_matrix.view(-1, 3, 3)

        h_matrix = h_matrix.view(-1, 3, 3).view(b_size * num_d, 1, 1, 3, 3)
        h_matrix = h_matrix.expand(b_size * num_d, h, w, 3, 3)
        h_matrix = h_matrix.contiguous().view(-1, 3, 3)
        warped_locs = torch.split(torch.bmm(h_matrix, self.xy_locations.to(device_)),
                                  split_size_or_sections=1,
                                  dim=1)
        x_locs = warped_locs[0] / warped_locs[2]
        x_locs = (x_locs - ((w - 1.0) / 2.0)) / ((w - 1.0) / 2.0)
        y_locs = warped_locs[1] / warped_locs[2]
        y_locs = (y_locs - ((h - 1.0) / 2.0)) / ((h - 1.0) / 2.0)
        grid = torch.cat([x_locs, y_locs], dim=2)
        # mul_plane_grid-> b, d, 2, h, w
        grid = grid.view(b_size, num_d, h, w, 2).permute(0, 1, 4, 2, 3)
        return grid

    def forward(self, h_mats, warped_alphas):
        h, w = warped_alphas.shape[-2:]
        # b, d, _, h, w = warped_alphas.shape
        mul_plane_grid = self.get_grid(h_mats, h, w)
        grid = self.alpha_composite(
            mul_plane_grid, warped_alphas).permute(0, 2, 3, 1)
        return grid


class WarpWithFlowFields(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.compute_flow_fields = ComputeFlowFields(configs)

    def forward(self, h_matrix, src_img, warped_alphas):
        b, f, h, w = src_img.shape
        grid = self.compute_flow_fields(h_matrix, warped_alphas)
        # print('compute_flow_fields grid shape', grid.shape)
        # print('compute_flow_fields src_image shape', src_img.shape)

        warped_views = F.grid_sample(input=src_img, grid=grid, mode='bilinear')
        return warped_views
