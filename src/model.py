from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Seperate(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc_sep = nn.Linear(17, 17)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, F, C = x.shape
        x = x.view(B, F, 17, -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = x.transpose(2, 3)
        x = self.fc_sep(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(2, 3)

        x = self.fc2(x)
        x = self.drop(x)

        x = x.view(B, F, C)
        return x

class RelativeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9, use_swin=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_swin = use_swin
        if not self.use_swin:
            self.B = nn.Parameter(torch.randn(1, num_heads, seq, seq), requires_grad=True)
        else:
            self.B = nn.Parameter(torch.zeros(1, num_heads, 2*seq - 1))
            trunc_normal_(self.B, std=0.02)
            coords_h = torch.arange(seq)
            coords_w = torch.arange(seq)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            B_index = coords[0, :, :] - coords[1, :, :]
            B_index = B_index + seq - 1
            self.register_buffer("B_index", B_index)
        


    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if not self.use_swin:
            bias = self.B
        else:
            bias = self.B[:, :, self.B_index.view(-1)].view(1, self.num_heads, N, N)

        attn = (q @ k.transpose(-2, -1)) * self.scale + bias

        if mask is not None:
            mask = mask.view(B, 1, N, N)
            attn = mask + attn

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., factor=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.view(B, 1, N, N)
            attn = mask + attn


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ExAttention(nn.Module):
    def __init__(self, dim, up_ratio=2, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.trans_dims = nn.Linear(dim, up_ratio * dim)

        self.linear_0 = nn.Linear(dim, up_ratio * dim)
        self.linear_1 = nn.Linear(dim, up_ratio * dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(up_ratio * dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seed):
        B, F, C = x.shape
        N = seed.shape[0]

        seed = seed.view(N, -1)

        x = self.trans_dims(x).view(B, F, self.heads, -1).permute(0, 2, 1, 3)  # B, H, F, C

        w0 = self.linear_0(seed).view(N, self.heads, -1).permute(1, 0, 2)  # (H, N, C)

        attn = torch.einsum('bhfc, hcn -> bhfn', x, w0.transpose(1, 2))
        # attn = x @ w0.view(1, self.heads, N, -1).repeat(B, 1, 1, 1).transpose(2, 3)  # (B, H, F, N)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-7 + attn.sum(dim=-1, keepdim=True))

        mean_attn = attn.mean(0).mean(0).mean(0)

        attn = self.attn_drop(attn)

        w1 = self.linear_1(seed).view(N, self.heads, -1).permute(1, 0, 2)  # (H, N, C)

        x = torch.einsum('bhfn, hnc -> bhfc', attn, w1)
        # x = attn @ w1.view(1, self.heads, N, -1).repeat(B, 1, 1, 1)

        x = x.permute(0, 2, 1, 3).reshape(B, F, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mean_attn

class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., drop_rate=0.0, norm_layer=nn.LayerNorm, seq=9, attention_type=Attention, use_swin=True):
        super().__init__()
        self.groups = ((1, 2, 3), (4, 5, 6), (0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16))
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.attn = nn.ModuleList([
            Block(dim * len(self.groups[i]), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop, norm_layer=norm_layer, attn_type=attention_type, seq=seq, use_swin=use_swin)
            for i in range(len(self.groups))
        ])

    def forward(self, x, mask=None):
        B, N, C = x.shape
        x = x.view(B, N, 17, -1)
        x_ = []
        for i in range(len(self.groups)):
            x_i = x[:, :, self.groups[i], :].view(B, N, -1)
            x_i = self.attn[i](x_i, mask)
            x_.append(x_i.view(*x_i.shape[:2], len(self.groups[i]), -1))
        x = torch.cat(x_, 2)
        x = x[:, :, self.inverse_group, :].view(B, N, -1)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type=Attention, seq=9, use_swin=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_type = attn_type
        if attn_type is Attention:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=seq, use_swin=use_swin)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if self.attn_type is RelativeAttention:
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ShiftGroupBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False, attention_type=Attention, use_swin=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GroupAttention(
            dim // 17, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, seq=M, attention_type=attention_type, use_swin=use_swin)
        self.norm2 = norm_layer(dim)
        self.shift_attn1 = GroupAttention(
            dim // 17, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, seq=M, attention_type=attention_type, use_swin=use_swin)
        self.norm3 = norm_layer(dim)
        self.shift_attn2 = GroupAttention(
            dim // 17, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, seq=M, attention_type=attention_type, use_swin=use_swin)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim//17 * mlp_ratio)
        self.norm4 = norm_layer(dim)
        self.mlp = Seperate(in_features=dim//17, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm5 = norm_layer(dim)
        self.shift_mlp1 = Seperate(in_features=dim//17, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm6 = norm_layer(dim)
        self.shift_mlp2 = Seperate(in_features=dim//17, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.M = M
        self.shift = shift
        self.downsample = downsample


    def shift_features(self, x, s):
        # x: (b, f, c)
        x = torch.cat([x[:, self.shift:, :], x[:, :self.shift, :]], 1)
        mask = torch.zeros(self.M, self.M, device=x.device, dtype=x.dtype)
        mask[:self.M - s, self.M - s:] = -np.inf
        mask[self.M - s:, :self.M - s] = -np.inf
        group_num = x.shape[1] // self.M
        mask_zeros = torch.zeros(group_num - 1, self.M, self.M, device=x.device, dtype=x.dtype)
        mask = torch.cat([mask_zeros, mask.view(1, self.M, self.M)], 0)
        return x, mask


    def forward(self, x):
        # x: (b, f, c)
        b, f, c = x.shape

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x, self.shift)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn1(self.norm2(x), mask))
        x = x + self.drop_path(self.shift_mlp1(self.norm5(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x, self.shift*2)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn2(self.norm3(x), mask))
        x = x + self.drop_path(self.shift_mlp2(self.norm6(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x = torch.cat([x[:, -self.shift*2:, :], x[:, :-self.shift*2, :]], 1)


        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return x

class ExtBaseBlock(nn.Module):
    def __init__(self, dim, up_ratio=2, num_heads=8, attn_drop=0., drop=0., drop_path=0.0, norm_layer=nn.LayerNorm, mlp_ratio=2.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ExAttention(dim, num_heads=num_heads, up_ratio=up_ratio, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seed):
        x_, attn = self.attn(self.norm1(x), seed)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class ExtBlock(nn.Module):
    def __init__(self, dim, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2.0, act_layer=nn.GELU, downsample=False):
        super().__init__()
        self.groups = ((1, 2, 3), (4, 5, 6), (0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16))
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)
        self.attn = nn.ModuleList([
            ExtBaseBlock((len(self.groups[i])) * dim // 17, attn_drop=attn_drop, drop=drop, norm_layer=norm_layer, act_layer=act_layer, mlp_ratio=mlp_ratio)
            for i in range(len(self.groups))])
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.downsample = downsample
        mlp_hidden_dim = int(dim // 17 * mlp_ratio)
        self.mlp = Seperate(in_features=dim//17, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm = norm_layer(dim)

    def forward(self, x, seed):
        # x: (b, f, c), seed: (N, 17, c)
        b, f, c = x.shape
        x_tmp = x.view(b, f, 17, -1)
        x_ = []
        mean_attn = []
        for i in range(len(self.groups)):
            x_i = x_tmp[:, :, self.groups[i][:], :].view(b, f, -1)
            seed_i = seed[:, self.groups[i][:], :]
            x_i, mean_attn_i = self.attn[i](x_i, seed_i)
            x_i = x_i.view(b, f, len(self.groups[i]), -1)
            # x_i = torch.cat([torch.zeros(b, f, 1, c // 17, device=x_i.device, dtype=torch.float32), x_i], 2)
            x_.append(x_i)
            mean_attn.append(mean_attn_i)
        x_ = torch.cat(x_, 2)
        # mean_attn = torch.stack(mean_attn, 0)
        mean_attn = torch.stack(mean_attn, 0).mean(0)
        x_ = x_[:, :, self.inverse_group, :].view(b, f, c)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm(x)))
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)

        return x, mean_attn
        
class Model(nn.Module):
    def __init__(self, cfg, seed=None):
        super().__init__()
        num_frame = cfg.number_of_frames
        num_joints = cfg.number_of_joints
        in_chans = 2
        embed_dim_ratio = 32
        num_heads = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        norm_layer = None
        M = 9
        shift = 3
        use_swin = cfg.swin_pos 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.seed_embed = nn.Linear(3, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        if seed is None:
            self.seed = nn.Parameter(torch.zeros(1024, 17, 3), requires_grad=True)
            trunc_normal_(self.seed, 0.02)
        else:
            self.seed = nn.Parameter(seed, requires_grad=False)
        ## The Spatial Aggregation Module
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        ## The PATA module
        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)
        down_sample = [False, False, False, False]
        for i in range(min(down_blocks, 4)):
            down_sample[i] = True
        self.blocks = nn.ModuleList([
            ShiftGroupBlock(
                dim=embed_dim, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                M=M, shift=shift, downsample=down_sample[i], attention_type=RelativeAttention, use_swin=use_swin)
            for i in range(2)])

        ## The PADA module
        self.ext_blocks = nn.ModuleList([
            ExtBlock(dim=embed_dim, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i-2], norm_layer=norm_layer,
                     mlp_ratio=mlp_ratio, downsample=down_sample[i])
            for i in range(2, 4)
        ])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.Ext_norm = norm_layer(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape
        x = rearrange(x, 'b f p c -> (b f) p c', )
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        return x

    def forward_features(self, x, ref_3D=None):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> b f p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, 'b f p c -> b f (p c)')
        seed = self.seed_embed(self.seed)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        mean_attn = 0.0
        for blk in self.ext_blocks:
            x, mean_attn_tmp = blk(x, seed)
            mean_attn = mean_attn + mean_attn_tmp
        x = self.Ext_norm(x)
        if ref_3D is not None:
            ref_3D = ref_3D.view(b, 17, 3)
            mean_attn = mean_attn / len(self.ext_blocks)
            _, idx = torch.topk(mean_attn, k=int(self.seed.shape[0] * 0.01), largest=False)
            idx = idx[:min(b, idx.shape[0])]
            # update according to the attention values
            self.seed.data[idx, :, :] = ref_3D[:idx.shape[0], :, :]
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, p, -1)
        return x

    def forward(self, x, ref_3D=None):
        # ref_3D: (b, 1, 17, 3), x: (b, f, 17, 2)
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.forward_features(x, ref_3D)
        x = self.Spatial_forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)
        return x
