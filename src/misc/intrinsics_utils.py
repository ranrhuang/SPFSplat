
import numpy as np
import torch
from einops import rearrange

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid




def estimate_focal_knowing_depth(pts3d, pp=None, focal_mode='weiszfeld', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    if pp is None:
        pp = torch.tensor((W/2, H/2), device=pts3d.device)

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # 1,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|

        z = pts3d[..., 2]  # Shape: (b, hw)
        valid_mask = z > 0  # Shape: (b, hw)


        pts3d = pts3d[valid_mask].unsqueeze(0)  # Shape: (valid_count, 3)
        pixels = pixels.expand(pts3d.shape[0], -1, -1)[valid_mask].unsqueeze(0)   


        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # (b, hw, 2), homogeneous (x,y,1)

        

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1) # (b, hw)
        dot_xy_xy = xy_over_z.square().sum(dim=-1) # (b, hw)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1) 


        if focal <= 0:
            print("init focal is less than zero", focal)
            focal = torch.full(focal.shape, focal_base, device=focal.device)
            

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)

    else:
        raise ValueError(f'bad {focal_mode=}')

    
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)

    if focal <= 0:
        print("iterated focal is less than zero", focal)
        focal = torch.full(focal.shape, focal_base, device=focal.device)

    return focal.ravel()


def convert_focal_to_intrinsics(estimated_focal, height, width): 
    '''
        estimated_focal: [..., 1]

        output: [..., 3, 3]
    '''
    
    c_x = width / 2.0
    c_y = height / 2.0

    focal_lengths = estimated_focal.squeeze(-1)  
    intrinsics = torch.zeros(*estimated_focal.shape[:-1], 3, 3).to(estimated_focal.device)

    # Fill the intrinsic matrices
    intrinsics[..., 0, 0] = focal_lengths  # f_x
    intrinsics[..., 1, 1] = focal_lengths  # f_y
    intrinsics[..., 0, 2] = c_x            # c_x (optical center x)
    intrinsics[..., 1, 2] = c_y            # c_y (optical center y)
    intrinsics[..., 2, 2] = 1.0            # homogeneous coordinate

    return intrinsics


def recover_intrinsics(intrinsics, width, height):
    '''
        intrinsics: [..., 3, 3]
        width: int
        height: int
    '''
    
    recovered_intrinsics = intrinsics.clone()
    recovered_intrinsics[..., 0, :] = intrinsics[..., 0, :] * width
    recovered_intrinsics[..., 1, :] = intrinsics[..., 1, :] * height

    return recovered_intrinsics

def normalize_intrinsics(intrinsics, width, height):
    '''
        intrinsics: [..., 3, 3]
        width: int
        height: int
    '''

    normalized_intrinsics = intrinsics.clone()
    normalized_intrinsics[..., 0, :] = intrinsics[..., 0, :] / width
    normalized_intrinsics[..., 1, :] = intrinsics[..., 1, :] / height

    return normalized_intrinsics



def estimate_intrinsics(pts3d, height, width):
    b, v = pts3d.shape[:2]
    focals = []
    for i in range(b):
        # focal = estimate_focal_knowing_depth(rearrange(pts3d[i,0][None], "b c h w -> b h w c")) 
        focal = estimate_focal_knowing_depth(pts3d[i,0][None]) 
        focals.append(focal)

    focals = torch.stack(focals)
    intrinsics = convert_focal_to_intrinsics(focals, height, width) 
    intrinsics = normalize_intrinsics(intrinsics, height, width) # (b, 3, 3)

    return intrinsics