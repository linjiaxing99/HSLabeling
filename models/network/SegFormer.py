# -*- coding:utf-8 -*-

# @Filename: DeeplabV3
# @Project : Glory
# @date    : 2024-12-28 21:52
# @Author  : Jaixing

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from models.network.resnet import *
from models.network.mit import *
from mmcv.cnn import ConvModule
from mmseg.ops import resize
''' 
-> SegFormer Head
'''

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, in_channels, in_index, channels, num_classes, embedding_dim=256, dropout_ratio=0.1, align_corners=False):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.input_transform = 'multiple_select'

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


'''
-> SegFormer
'''

class SegFormer(nn.Module):
    def __init__(self,
                 backbone,
                 feature_strides=None,
                 in_channels=None,
                 in_index=None,
                 channels=128,
                 num_classes=5,
                 embedding_dim=256,
                 freeze_bn=False):

        super(SegFormer, self).__init__()

        if in_index is None:
            in_index = [0, 1, 2, 3]
        if in_channels is None:
            in_channels = [32, 64, 160, 256]
        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]

        if 'mit_b0' in backbone:
            self.backbone = mit_b0()
            checkpoint = torch.load('/home/isalab206/Downloads/pretrained/mit_b0.pth', map_location=torch.device('cpu'))
            self.backbone.load_state_dict(checkpoint)
            print('load pretrained transformer')

        elif 'mit_b3' in backbone:
            self.backbone = mit_b3()
            checkpoint = torch.load('/home/isalab206/Downloads/pretrained/mit_b3.pth', map_location=torch.device('cpu'))
            self.backbone.load_state_dict(checkpoint)
            print('load pretrained transformer')

        self.decoder = SegFormerHead(feature_strides, in_channels, in_index, channels, num_classes, embedding_dim)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone(x)
        feat = x[3]

        x = self.decoder(x)
        # x = F.interpolate(x, size=(h, w), mode='bilinear')

        return x, feat

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':

    model = SegFormer(num_classes=4)

    x = torch.rand(2, 4, 256, 256)
    x = model(x)

    print(x.shape)