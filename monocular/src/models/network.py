"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock, ResBlock, DeconvBlock

class BaseEncoderDecoder(torch.nn.Module):
    """ This class serves as the base encoder decoder network usef for estimating"""

    def __init__(self, in_chans, feat_size=32, output_feats=96, kernel_size=3):
        super(BaseEncoderDecoder, self).__init__()
        # self.scale = scale
        # self.conv_block_0 = ConvBlock(in_chans, feat_size, 7)  # H
        self.conv_block_1 = ConvBlock(in_chans, feat_size, 5)  # H/2
        self.conv_block_2 = ConvBlock(feat_size, int(2 * feat_size), 5)  # H/4
        self.conv_block_3 = ConvBlock(int(2 * feat_size), int(4 * feat_size), kernel_size)  # H/8
        self.conv_block_4 = ConvBlock(int(4 * feat_size), int(8 * feat_size), kernel_size)  # H/16
        self.conv_block_5 = ConvBlock(int(8 * feat_size), int(16 * feat_size), kernel_size)  # H/32
        self.conv_block_6 = ConvBlock(int(16 * feat_size), int(16 * feat_size), kernel_size)  # H/64
        self.conv_block_7 = ConvBlock(int(16 * feat_size), int(16 * feat_size), kernel_size)  # H/128
        # input from   [conv_block_7]
        self.up_conv_7 = DeconvBlock(int(16 * feat_size), int(16 * feat_size))          # H/64
        # input from   [up_conv7, conv_block_6]
        self.up_conv_6 = DeconvBlock(int(32 * feat_size), int(16 * feat_size))          # H/8
        # input from   [up_conv6, conv_block_5]
        self.up_conv_5 = DeconvBlock(int(32 * feat_size), int(16 * feat_size))          # H/8
        # input from   [up_conv5, conv_block_4]
        self.up_conv_4 = DeconvBlock(int(24 * feat_size), int(12 * feat_size))          # H/8
        # input from   [up_conv4, conv_block_3]
        self.up_conv_3 = DeconvBlock(int(16 * feat_size), int(8 * feat_size))           # H/8
        # input from   [up_conv3, conv_block_2]
        self.up_conv_2 = DeconvBlock(int(10 * feat_size), int(3 * feat_size))           # H/8
        # input from  [up_conv2, conv_block_1]
        self.up_conv_1 = DeconvBlock(int(4 * feat_size), int(3 * feat_size))            # H/4
        self.out_conv = ConvBlock(int(3 * feat_size), output_feats, 3,
                                        down_sample=False)            # H/2

    def forward(self, input):
        b_1 = self.conv_block_1(input)
        b_2 = self.conv_block_2(b_1)
        b_3 = self.conv_block_3(b_2)
        b_4 = self.conv_block_4(b_3)
        b_5 = self.conv_block_5(b_4)
        b_6 = self.conv_block_6(b_5)
        b_7 = self.conv_block_7(b_6)

        u_7 = self.up_conv_7(b_7)
        u_6 = self.up_conv_6([u_7, b_6])
        u_5 = self.up_conv_5([u_6, b_5])
        u_4 = self.up_conv_4([u_5, b_4])
        u_3 = self.up_conv_3([u_4, b_3])
        u_2 = self.up_conv_2([u_3, b_2])
        u_1 = self.up_conv_1([u_2, b_1])
        out = self.out_conv(u_1)
        return out


class ConvNetwork(torch.nn.Module):

    def __init__(self, configs):
        super(ConvNetwork, self).__init__()
        self.configs = configs

        input_channels = 3
        num_planes = configs['num_planes']
        enc_features = configs['encoder_features']
        encoder_ouput_features = configs['encoder_ouput_features']
        self.input_channels = input_channels
        self.num_planes = num_planes
        # if output contains color
        self.blending_w_chans = 1
        self.out_bg_chans = 3
        use_no_relu = True
        self.discriptor_net = BaseEncoderDecoder(in_chans=configs['input_channels'], feat_size=enc_features, output_feats=encoder_ouput_features, kernel_size=3)
        # alpha, blending_w, background, and segmentation
        self.base_res_layers = nn.Sequential(*[ResBlock(encoder_ouput_features, 3) for i in range(2)])
        self.blending_w_alpha_pred = nn.Sequential(ResBlock(encoder_ouput_features, 3),
                                ResBlock(encoder_ouput_features, 3),
                                nn.BatchNorm2d(encoder_ouput_features),
                                ConvBlock(encoder_ouput_features, num_planes, 3, down_sample=False),
                                nn.BatchNorm2d(num_planes),
                                ConvBlock(num_planes, num_planes*2, 3, down_sample=False, use_no_relu=True))
        self.bg_pred = nn.Sequential(ResBlock(encoder_ouput_features, 3),
                                ResBlock(encoder_ouput_features, 3),
                                ConvBlock(encoder_ouput_features, configs['out_put_channels'], 3, down_sample=False, use_no_relu=True))

    def forward(self, input_img):
        # assert self.configs['source_code'][1]=='1', 'semantic mpi expects source_code 01x'
        b, nc, h, w = input_img.shape
        blend_chans = self.blending_w_chans
        bg_chans = self.out_bg_chans
        feats_0 = self.discriptor_net(input_img)
        feats_1 = self.base_res_layers(feats_0)
        blending_alpha = self.blending_w_alpha_pred(feats_1)
        blending_alpha = blending_alpha.view(b, self.num_planes, 2, h, w)
        blending_weights = torch.sigmoid(blending_alpha[:, :, 0, :, :])
        alpha = blending_alpha[:, :, 1, :, :]
        bg_img = torch.sigmoid(self.bg_pred(feats_0))
        alpha, blending_weights = alpha.unsqueeze(2), blending_weights.unsqueeze(2)
        return alpha, blending_weights, bg_img
