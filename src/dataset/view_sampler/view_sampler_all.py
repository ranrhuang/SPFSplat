from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerAllCfg:
    name: Literal["all"]


class ViewSamplerAll(ViewSampler[ViewSamplerAllCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        v, _, _ = extrinsics.shape
        all_frames = torch.arange(v, device=device)
        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return all_frames, all_frames, overlap

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
