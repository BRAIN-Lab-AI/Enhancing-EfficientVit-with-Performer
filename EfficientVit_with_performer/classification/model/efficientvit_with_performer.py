
# --------------------------------------------------------
# EfficientViT Model Architecture (with Performer Attention)
# --------------------------------------------------------
import torch
#import itertools
from performer_pytorch import SelfAttention as PerformerAttention
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class PerformerWrapper(torch.nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.attn = PerformerAttention(
            dim=dim,
            heads=heads,
            causal=False,
            dropout=dropout
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.attn(x)  # B, N, C
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class EfficientViTBlock(torch.nn.Module):    
    def __init__(self, type, ed, kd, nh=8, ar=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(PerformerWrapper(ed, nh))  # Replaced attention

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 stages=['s', 's', 's'], embed_dim=[64, 128, 192], key_dim=[16, 16, 16],
                 depth=[1, 2, 3], num_heads=[4, 4, 4], window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5], down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False):
        super().__init__()

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8)
        )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        self.blocks1, self.blocks2, self.blocks3 = [], [], []

        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, resolution=resolution)),
                    Residual(FFN(ed, int(ed * 2), resolution)),
                ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),
                ))

        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
