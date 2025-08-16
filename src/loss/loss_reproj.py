
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat, einsum
import numpy as np


from ..misc.cam_utils import project_to_cam
from .loss import Loss

@dataclass
class LossReprojCfg:
    weight: float
    mode: str
    circle_schedule: bool
    total_iterations: int


@dataclass
class LossReprojCfgWrapper:
    reproj: LossReprojCfg


def weighted_tanh(repro_errs, weight):
    return weight * torch.tanh(repro_errs / weight).sum()

class LossReproj(Loss[LossReprojCfg, LossReprojCfgWrapper]):
    # Modified from https://github.com/nianticlabs/acezero/blob/main/ace_loss.py and https://github.com/nianticlabs/acezero/blob/main/ace_trainer.py
    def __init__(self, cfg: LossReprojCfgWrapper) -> None:
        super().__init__(cfg)

        """
        Compute per-pixel reprojection loss using different configurable approaches.

        - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                    error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
        - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                    the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                    is True (default), in which case it applies a circular scheduling. See paper for details.
        - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
        - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                    `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
        - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
        """

        self.repro_loss_hard_clamp = 1000
        self.soft_clamp = 50
        self.soft_clamp_min = 1
    

    def forward(self, pts3d, im_poses, intrinsics, global_step, detach_pts3d=False):
        '''
            pts3d : [b, h, w, 3]
            im_poses : [b, 4, 4]
            intrinsics : [b, 3, 3]
        '''
        device = pts3d.device
        b, height, width, _ = pts3d.shape

        if detach_pts3d:
            pts3d = pts3d.detach()
       
        unnorm_intrinsics = intrinsics.clone()
        unnorm_intrinsics[..., 0, :] = intrinsics[..., 0, :] * width
        unnorm_intrinsics[..., 1, :] = intrinsics[..., 1, :] * height
        
        pred_px = project_to_cam(rearrange(pts3d, "b h w c -> b (h w) c"), im_poses, unnorm_intrinsics, return_z=False) # (b, n, 2)

        pred_px = rearrange(pred_px, "b (h w) c -> b h w c", h=height, w=width)
        

        origin = [0,0]
        tw, th = [torch.arange(o, o + s) for s, o in zip((width, height), origin)]
        grid = torch.meshgrid(tw, th, indexing='xy')
        pixel_grid_2HW = torch.stack(grid) # [2, h, w]
        pixel_positions_12HW = pixel_grid_2HW[None]  # [1, 2, h, w]
        target_pixel_positions_B2HW = pixel_positions_12HW.repeat(b, 1, 1, 1)  # [b, 2, h, w]
        target_px = rearrange(target_pixel_positions_B2HW, "b c h w -> b h w c").to(device) # [b, h, w, 2]

        reprojection_error = pred_px - target_px # [b, h, w, 2]
        reprojection_error = torch.norm(reprojection_error, dim=-1, keepdim=True, p=2) # [b, h, w, 1]

        
        invalid_repro = reprojection_error > self.repro_loss_hard_clamp
        invalid_mask = invalid_repro
        valid_mask = ~invalid_mask


        if valid_mask.sum() > 0:
            # Reprojection error for all valid scene coordinates.
            valid_reprojection_error = reprojection_error[valid_mask] # [valid_num]

            # Compute the loss for valid predictions.
            repro_loss_valid = self.compute(valid_reprojection_error, global_step)
            repro_loss_valid = repro_loss_valid / valid_reprojection_error.shape[0]
        else:
            repro_loss_valid = 0

        return repro_loss_valid

    def compute(self, repro_errs_b1N, iteration):
        """
        Compute the reprojection loss based on the type of loss function specified during the initialization of the class.
        The types of loss function available are: 'tanh', 'dyntanh', 'l1', 'l1+sqrt', and 'l1+logl1'.

        :param repro_errs_b1N: A tensor containing the reprojection errors.
        :param iteration: The current iteration of the training process.
        :return: The computed loss as a scalar.
        """
        # If there are no elements in the reprojection errors tensor, return 0
        if repro_errs_b1N.nelement() == 0:
            return 0

        # Compute the simple tanh loss
        if self.cfg.mode == "tanh":
            return self.cfg.weight * weighted_tanh(repro_errs_b1N, self.soft_clamp)

        # Compute the dynamic tanh loss
        elif self.cfg.mode == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.cfg.total_iterations

            # Optionally scale it using the circular schedule.
            if self.cfg.circle_schedule:
                schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return self.cfg.weight * weighted_tanh(repro_errs_b1N, loss_weight)

        # Compute the L1 loss
        elif self.cfg.mode == "l1":
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            return self.cfg.weight * repro_errs_b1N[~softclamp_mask_b1].sum()

        # Compute the L1 loss for small errors and sqrt loss for larger errors
        elif self.cfg.mode == "l1+sqrt":
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * repro_errs_b1N[softclamp_mask_b1]).sum()

            return self.cfg.weight * (loss_l1 + loss_sqrt)

        # Compute the L1 loss for small errors and log L1 loss for larger errors
        else:
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * repro_errs_b1N[softclamp_mask_b1])).sum()

            return self.cfg.weight * (loss_l1 + loss_logl1)
        
