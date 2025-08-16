from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from einops import rearrange

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.mm(R1.T, R2)) - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.acos(cos)))


def angle_error_vec(v1, v2):
    n = torch.norm(v1) * torch.norm(v2)
    cos_theta = torch.dot(v1, v2) / (n + 1e-9)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.acos(cos_theta))


def compute_translation_error(t1, t2):
    return torch.norm(t1 - t2)


@torch.no_grad()
def compute_pose_error(pose_gt, pose_pred):
    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_pred[:3, :3]
    t = pose_pred[:3, 3]

    error_t = angle_error_vec(t, t_gt)
    error_t = torch.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_t_scale = compute_translation_error(t, t_gt)
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_t_scale, error_R


@torch.no_grad()
def compute_pose_error_for_batch(pred_pose, tgt_pose):
    # for relative estimated pose
    '''
        pred_pose : [b, v, 4, 4] or [b, 4, 4] or [4, 4]
        tgt_pose: [b, v, 4, 4] or [b, 4, 4] or [4, 4]
    '''

    rel_angular_error = 0
    rel_transl_error = 0

    if pred_pose.ndimension() == 4:
        pred_pose = rearrange(pred_pose, "b v ... -> (b v) ...")
        tgt_pose = rearrange(tgt_pose, "b v ... -> (b v) ...")
    if pred_pose.ndimension() == 2:
        pred_pose = pred_pose.unsqueeze(0)
        tgt_pose = tgt_pose.unsqueeze(0)

    cnt = pred_pose.shape[0]

    for i in range(cnt):
        estimated_transl_error, _, estimated_angular_error = compute_pose_error(tgt_pose[i].cpu(), pred_pose[i].cpu()) # 
        rel_angular_error += estimated_angular_error
        rel_transl_error += estimated_transl_error

    rel_angular_error = rel_angular_error / cnt
    rel_transl_error = rel_transl_error /  cnt
    return rel_angular_error, rel_transl_error
