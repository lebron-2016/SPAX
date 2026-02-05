import torch
from torch import nn
from einops.layers.torch import Rearrange
from sparsecnn.sparse_layers import SPConvTranspose2d, SPModule, SPConv2d, SPBatchNorm2d, SPActivation, SPSparsify

class SpatialAttention(SPModule):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = SPConv2d(2, 1, 5, padding=2, bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1).contiguous(memory_format=torch.channels_last)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x*SPModule.mask_backup[x.shape[2]])
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(SPModule):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = SPConv2d(dim, dim, 5, padding=2, groups=dim, bias=True) # 2 * dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        # x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = x + pattn1
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2).contiguous(memory_format=torch.channels_last)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2