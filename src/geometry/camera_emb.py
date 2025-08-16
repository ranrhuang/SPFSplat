from einops import rearrange
import torch
import math

from .projection import sample_image_grid, get_local_rays, get_cam_xy, get_world_rays
from ..misc.sht import rsh_cart_2, rsh_cart_4, rsh_cart_6, rsh_cart_8


def get_intrinsic_embedding(context, degree=0, downsample=1, merge_hw=False):
    assert degree in [0, 2, 4, 8]

    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_local_rays(xy_ray, rearrange(context["intrinsics"], "b v i j -> b v () () i j"),)

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "b v h w d -> b v (h w) d")
    else:
        directions = rearrange(directions, "b v h w d -> b v d h w")

    return directions


def get_intrinsic_positional_embedding(context, d_model, downsample=1, ):
    b, v, _, h, w = context["image"].shape
    device = context["image"].device

    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_cam_xy(xy_ray, rearrange(context["intrinsics"], "b v i j -> b v () () i j"),) # (b, v, h, w, 2)
    # print("directions", directions.shape, directions)
    x_position = directions[..., 0].unsqueeze(-1)
    y_position = directions[..., 1].unsqueeze(-1)
    # z_position = directions[..., 2]
    # print("x_position", x_position.shape)


    div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2))).to(device)
    div_term = div_term[None, None, None, None, :]
    # print("div_term", div_term.shape)

    pe = torch.zeros((b, v, h//downsample, w//downsample, d_model)).to(device)

    pe[:, :, :, :, 0::4] = torch.sin(x_position * div_term) 
    pe[:, :, :, :, 1::4] = torch.cos(x_position * div_term)
    pe[:, :, :, :, 2::4] = torch.sin(y_position * div_term)
    pe[:, :, :, :, 3::4] = torch.cos(y_position * div_term)

    # pe_downsample = pe[:, :, 0::downsample, 0::downsample]
    # print("pe", pe.shape)


    return pe

def get_plucker_embedding(context, extrinsics, downsample=1):
    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    

    origins, directions = get_world_rays(xy_ray, rearrange(extrinsics, "b v i j -> b v () () i j"), rearrange(context["intrinsics"], "b v i j -> b v () () i j") ) #  # (b, v, h, w, 3)

    dray_pluc = torch.cat([torch.cross(origins, directions, dim=-1), directions], dim=-1).permute(0, 1, 4, 2, 3) # [b v h w 6]   
    # print("dray_pluc", dray_pluc.shape)     

    return dray_pluc