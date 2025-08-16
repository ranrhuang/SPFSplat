from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    apply_after_step: int

@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


# class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
#     def forward(
#         self,
#         prediction: DecoderOutput,
#         batch: BatchedExample,
#         gaussians: Gaussians,
#         global_step: int,
#     ) -> Float[Tensor, ""]:
#         delta = prediction.color - batch["target"]["image"]
#         return self.cfg.weight * (delta**2).mean()


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: Tensor,
        image: Tensor,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        
        delta = prediction - image # (b, v, 3, h, w)
       

        return self.cfg.weight * (delta**2).mean()
