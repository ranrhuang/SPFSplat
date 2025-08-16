from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    enable_cov_grad: bool
    enable_sh_grad: bool

class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        # dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.enable_cov_grad = cfg.enable_cov_grad
        self.enable_sh_grad = cfg.enable_sh_grad
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
            repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
            scale_invariant=self.make_scale_invariant,
            enable_cov_grad=self.enable_cov_grad,
            enable_sh_grad=self.enable_sh_grad
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        
        depth = rearrange(depth, "(b v) 1 h w -> b v h w", b=b, v=v)

        if self.make_scale_invariant:
            scale = near / 1
            depth = depth * scale[:, :, None, None]

        return DecoderOutput(color, depth)
        


  