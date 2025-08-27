import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res



class Decoder(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout
        self.args = args

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        
        if "LA" in args.dataset:
            self.attn_x5 = CA_TransformerDecoderLayer(input_size=[7, 7, 5], in_channels=256, d_model=512, patch_size=[7, 7, 5])
        else:
            self.attn_x5 = CA_TransformerDecoderLayer(input_size=[6, 6, 6], in_channels=256, d_model=512, patch_size=[6, 6, 6])
        
        self.projection_head = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=1)

    def forward(self, features, txt_embed):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5 = self.attn_x5(x5, txt_embed) + x5
        feats_x5 = self.projection_head(x5)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg, feats_x5


class Decoder_DTC(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder_DTC, self).__init__()
        self.has_dropout = has_dropout
        self.args = args

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        
        if "LA" in args.dataset:
            self.attn_x5 = CA_TransformerDecoderLayer(input_size=[7, 7, 5], in_channels=256, d_model=512, patch_size=[7, 7, 5])
        else:
            self.attn_x5 = CA_TransformerDecoderLayer(input_size=[6, 6, 6], in_channels=256, d_model=512, patch_size=[6, 6, 6])
        
        self.projection_head = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=1)

    def forward(self, features, txt_embed):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5 = self.attn_x5(x5, txt_embed) + x5
        feats_x5 = self.projection_head(x5)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        # out_seg = self.out_conv(x9)

        return x9, feats_x5


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=7):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)

        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        project_out = not (num_heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, p1, p2):
        q_p1 = self.w_q(p1)
        k_p2 = self.w_k(p2)
        v_p2 = self.w_v(p2)
        q_p1 = rearrange(q_p1, 'b n (h d) -> b h n d', h=self.num_heads)
        k_p2 = rearrange(k_p2, 'b n (h d) -> b h n d', h=self.num_heads)
        v_p2 = rearrange(v_p2, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_p1p2 = einsum('b h i d, b h j d -> b h i j', q_p1, k_p2) * self.scale
        attn_p1p2 = attn_p1p2.softmax(dim=-1)

        attn_p1p2 = einsum('b h i j, b h j d -> b h j d', attn_p1p2, v_p2)
        attn_p1p2 = rearrange(attn_p1p2, 'b h n d -> b n (h d)')

        attn_p1p2 = self.to_out(attn_p1p2)
        return attn_p1p2


class CA_TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            input_size,
            in_channels,
            d_model,

            nhead=8,
            dropout=0.1,
            patch_size=[7, 7, 5],
    ):
        super().__init__()

        # patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding_Axial = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[1], p2=patch_size[2]),
            nn.Linear(in_channels * patch_size[1] * patch_size[2], d_model)
        )
        self.to_patch_embedding_Sagittal = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[2]),
            nn.Linear(in_channels * patch_size[0] * patch_size[2], d_model)
        )
        self.to_patch_embedding_Coronal = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(in_channels * patch_size[0] * patch_size[1], d_model)
        )
        
        # Positional encodings for Axial, Sagittal, Coronal views
        self.positional_encoding_Axial = nn.Parameter(torch.randn(1, (input_size[1] // patch_size[1]) * (input_size[2] // patch_size[2]), d_model))
        self.positional_encoding_Sagittal = nn.Parameter(torch.randn(1, (input_size[0] // patch_size[0]) * (input_size[2] // patch_size[2]), d_model))
        self.positional_encoding_Coronal = nn.Parameter(torch.randn(1, (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1]), d_model))
        
        self.view_weights = nn.Parameter(torch.ones(3))

        self.img_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.img_txt_attn = CA_Attention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.to_out_Axial = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[1] * patch_size[2]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[1] // patch_size[1]),
                      w=(input_size[2] // patch_size[2]), p1=patch_size[1], p2=patch_size[2]),
        )
        self.to_out_Sagittal = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[0] * patch_size[2]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[0] // patch_size[0]),
                      w=(input_size[2] // patch_size[2]), p1=patch_size[0], p2=patch_size[2]),
        )
        self.to_out_Coronal = nn.Sequential(
            nn.Linear(d_model, in_channels * patch_size[0] * patch_size[1]),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size[0] // patch_size[0]),
                      w=(input_size[1] // patch_size[1]), p1=patch_size[0], p2=patch_size[1]),
        )


        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1)

    def cross_attn(self, x, text_feature, view):
        
        if view == "Axial":
            q = self.to_patch_embedding_Axial(x)
            q = q + self.positional_encoding_Axial
        elif view == "Sagittal":
            q = self.to_patch_embedding_Sagittal(x)
            q = q + self.positional_encoding_Sagittal
        elif view == "Coronal":
            q = self.to_patch_embedding_Coronal(x)
            q = q + self.positional_encoding_Coronal

            
        q = k = v = self.norm1(q)
        q = q + self.img_attn(q, k, v)
        q = self.norm2(q)


        x = q + self.img_txt_attn(text_feature, q)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        if view == "Axial":
            x = self.to_out_Axial(x)
        elif view == "Sagittal":
            x = self.to_out_Sagittal(x)
        elif view == "Coronal":
            x = self.to_out_Coronal(x)
        return x, q

    def forward(self, img_feats, txt_emebd):
        # diff views
        # Axial_view = img_feats.mean(dim=2)
        # Sagittal_view = img_feats.mean(dim=3)
        # Coronal_view = img_feats.mean(dim=4)
        Axial_view = F.adaptive_avg_pool3d(img_feats, (1, img_feats.size(3), img_feats.size(4))).squeeze(2)
        Sagittal_view = F.adaptive_avg_pool3d(img_feats, (img_feats.size(2), 1, img_feats.size(4))).squeeze(3)
        Coronal_view = F.adaptive_avg_pool3d(img_feats, (img_feats.size(2), img_feats.size(3), 1)).squeeze(4)


        # views attn
        Axial_view_attn_map, _ = self.cross_attn(Axial_view, txt_emebd, view="Axial")
        Sagittal_view_attn_map, _ = self.cross_attn(Sagittal_view, txt_emebd, view="Sagittal")
        Coronal_view_attn_map, _ = self.cross_attn(Coronal_view, txt_emebd, view="Coronal")

        # view_3D = Axial_view_attn_map.unsqueeze(2) + Sagittal_view_attn_map.unsqueeze(3) + Coronal_view_attn_map.unsqueeze(4)
        view_3D = (
                    self.view_weights[0] * Axial_view_attn_map.unsqueeze(2) +
                    self.view_weights[1] * Sagittal_view_attn_map.unsqueeze(3) +
                    self.view_weights[2] * Coronal_view_attn_map.unsqueeze(4)
                )
        
        return img_feats + view_3D


class MC_Clip_3D(nn.Module):
    def __init__(self, args, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True, has_residual=False):
        super(MC_Clip_3D, self).__init__()

        self.args = args
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)
        self.decoder2 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        

    def forward(self, input, txt_embed):

        b = input.shape[0]
        if b > 1:
            txt_embed = txt_embed.repeat(b, 1, 1)
        else:
            txt_embed = txt_embed

        features = self.encoder(input)
        out_seg1, feats_1 = self.decoder1(features, txt_embed)
        out_seg2, feats_2 = self.decoder2(features, txt_embed)

        return [out_seg1, out_seg2], [feats_1, feats_2]
    

class MC_plus_Clip_3D(nn.Module):
    def __init__(self, args, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True, has_residual=False):
        super(MC_plus_Clip_3D, self).__init__()

        self.args = args
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder0 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)
        self.decoder1 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        self.decoder2 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=2)
        

    def forward(self, input, txt_embed):

        b = input.shape[0]
        if b > 1:
            txt_embed = txt_embed.repeat(b, 1, 1)
        else:
            txt_embed = txt_embed

        features = self.encoder(input)
        out_seg_0, feats_0 = self.decoder0(features, txt_embed)
        out_seg_1, feats_1 = self.decoder1(features, txt_embed)
        out_seg_2, feats_2 = self.decoder2(features, txt_embed)

        return [out_seg_0, out_seg_1, out_seg_2], [feats_0, feats_1, feats_2]
    
    
class vnet_clip_3D(nn.Module):
    def __init__(self, args, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True, has_residual=False):
        super(vnet_clip_3D, self).__init__()

        self.args = args
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)
        # self.decoder2 = Decoder(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        

    def forward(self, input, txt_embed):

        b = input.shape[0]
        if b > 1:
            txt_embed = txt_embed.repeat(b, 1, 1)
        else:
            txt_embed = txt_embed

        features = self.encoder(input)
        out_seg1, feats_1 = self.decoder1(features, txt_embed)
        # out_seg2, feats_2 = self.decoder2(features, txt_embed)

        return out_seg1, feats_1
    
    
class VNet_DTC(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet_DTC, self).__init__()
        self.has_dropout = has_dropout

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder_DTC(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)


    def forward(self, input, txt_embed):
        
        b = input.shape[0]
        if b > 1:
            txt_embed = txt_embed.repeat(b, 1, 1)
        else:
            txt_embed = txt_embed
            
        features = self.encoder(input)
        x9, feats = self.decoder(features, txt_embed)
        
        out = self.out_conv(x9)
        out_tanh = self.tanh(out)
        out_seg = self.out_conv2(x9)
        return [out_tanh, out_seg], feats