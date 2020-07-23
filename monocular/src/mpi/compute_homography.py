"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import torch
import torch.nn.functional as F

class ComputeHomography:
    '''Computes planar homography matrices for the mpi representation. Unlike other classes in this project this is not a nn.Module type,
    because there is not need to back-prop through this class/object
    :param configs: configuration dictionary
    '''
    def __init__(self, configs):
        self.configs = configs
        self.h, self.w = configs['height'], configs['width']
        self.num_depths = configs['num_planes']
        self.depth_proposals = 1 / torch.linspace(1 / configs['near_plane'], 1 / configs['far_plane'], configs['num_planes'])
        self.depth_proposals = self.depth_proposals.view(configs['num_planes']).float()
        self.src_corner_pts = [torch.Tensor([(self.w - 1) * i, (self.h - 1) * j, 1]) for i in range(2) for j in range(2)]

    def __call__(self, kmats, rmats, tvecs):
        '''
        :param kmats: batch of 3x3 camera intriciscs
        :param rmats: batch of rotation matrices
        :param t: batch of 3x1 translation vectors
        '''
        device_ = kmats.device
        batch_size = rmats.shape[0]
        num_dep = self.num_depths
        rmats = rmats.view(batch_size, 1, 3, 3).expand(batch_size, num_dep, 3, 3)
        rmats = rmats.contiguous().view(-1, 3, 3)
        tvecs = tvecs.view(batch_size, 1, 3, 1).contiguous().expand(batch_size, num_dep, 3, 1)
        tvecs = tvecs.contiguous().view(-1, 3, 1)

        kinv = torch.stack([torch.inverse(k) for k in kmats])
        kmats = kmats.view(-1, 1, 3, 3).expand(batch_size, num_dep, 3, 3).contiguous()
        kinv = kinv.view(-1, 1, 3, 3).expand(batch_size, num_dep, 3, 3).contiguous()
        kinv, kmats = kinv.view(-1, 3, 3), kmats.view(-1, 3, 3)
        n = torch.Tensor([0, 0, 1]).view(1, 1, 3).expand(rmats.shape[0], 1, 3)
        n = n.to(device_).float()
        depth_proposals = self.depth_proposals.view(1, num_dep, 1).to(device_)
        depth_proposals = depth_proposals.expand(batch_size, num_dep, 1).contiguous()
        depth_proposals = depth_proposals.view(-1, 1, 1)
        num_1 = torch.bmm(torch.bmm(torch.bmm(rmats.permute(0, 2, 1), tvecs), n), rmats.permute(0, 2, 1))
        den_1 = -depth_proposals - torch.bmm(torch.bmm(n, rmats.permute(0, 2, 1)), tvecs) 
        assert not(torch.isnan(den_1).any() or torch.isinf(den_1).any()), "nan found in den_1"

        h_mats = torch.bmm(torch.bmm(kmats, (rmats.permute(0, 2, 1) + (num_1 / den_1))), kinv)
        h_mats = h_mats.view(batch_size, num_dep, 3, 3)
        return h_mats
