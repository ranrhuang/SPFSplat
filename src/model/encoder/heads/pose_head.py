# --------------------------------------------------------
# pose head implementation 
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass

import math
import torch.nn.functional as F 
from itertools import repeat


@dataclass
class PoseHeadCfg:
    pose_init_t: bool
    use_homogeneous: bool
    concat_enc: bool


class PoseHead(nn.Module):
    cfg: PoseHeadCfg
    # modified from https://github.com/nianticlabs/marepo/blob/main/marepo/marepo_network.py
    def __init__(self, net, cfg: PoseHeadCfg):
        super().__init__()
        self.cfg = cfg

        if self.cfg.concat_enc:
            self.d_model = net.enc_embed_dim + net.dec_embed_dim
        else:
            self.d_model = net.dec_embed_dim

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

        self.more_mlps = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),  # Adjusted input size to include initial t and rot
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU()
        )
        
        self.use_homogeneous = self.cfg.use_homogeneous
        self.concat_enc = self.cfg.concat_enc

        # Fully connected layers for translation regression
        if self.cfg.use_homogeneous:
            self.fc_t = nn.Linear(self.d_model//4, 4)
            homogeneous_min_scale = 0.01
            homogeneous_max_scale = 4.0
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc_t = nn.Linear(self.d_model//4, 3)

        # Fully connected layers for rotation regression
        self.fc_rot = nn.Linear(self.d_model // 4, 6)

        self.init_pose()

    def init_pose(self):
        """Initialize weights and biases to output zero by default."""
 
        nn.init.constant_(self.fc_rot.weight, 0.0)
        nn.init.constant_(self.fc_rot.bias, 0.0)
        self.fc_rot.bias.data[:6] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32) # for identity matrix

        if self.cfg.pose_init_t:
            nn.init.constant_(self.fc_t.weight, 0.0)
            nn.init.constant_(self.fc_t.bias, 0.0)

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, conf=None):

        if self.cfg.concat_enc:
            enc_output, dec_output = encoder_tokens[0], encoder_tokens[-1] 
            cat_output = torch.cat([enc_output, dec_output], dim=-1)  

            p = int(math.sqrt(enc_output.shape[1]))
            feat = rearrange(cat_output, "b (h w) c -> b c h w", h=p, w=p)
        else:
            dec_output = encoder_tokens[-1]
            p = int(math.sqrt(dec_output.shape[1]))
            feat = rearrange(dec_output, "b (h w) c -> b c h w", h=p, w=p)
        

        if feat.shape[-1] > 1:
            feat = self.avgpool(feat)  # [b, c, 1, 1]
        feat = feat.view(feat.size(0), -1)  # Flatten to [b, c]

        # Pass through MLP layers
        feat = self.more_mlps(feat) # (b, c)
        out_t = self.fc_t(feat)  # [B, 4] or [B, 3] based on use_homogeneous

        if self.cfg.use_homogeneous:
            # Softplus ensures smooth homogeneous parameter adjustment
            h_slice = F.softplus(out_t[:, 3].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            out_t = out_t[:, :3] / h_slice  # Normalize translation

        # Compute rotation
        out_r = self.fc_rot(feat)  # [b, 6]


        out = torch.cat([out_r, out_t], dim=-1)

        return out


def create_pose_head(net, cfg):
    return PoseHead(net, cfg)



