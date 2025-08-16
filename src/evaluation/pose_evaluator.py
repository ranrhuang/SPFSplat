import json
import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.cam_utils import camera_normalization, pose_auc, update_pose, get_pnp_pose

import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from ..loss.loss_ssim import ssim
from ..misc.image_io import load_image, save_image
from ..misc.utils import inverse_normalize, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim, compute_pose_error
from ..misc.intrinsics_utils import estimate_intrinsics
from ..global_cfg import get_cfg
from ..evaluation.metrics import compute_pose_error_for_batch

class PoseEvaluator(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.cfg = cfg

        # our model
        self.encoder = encoder.to(self.device)
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)

        self.data_shim = get_data_shim(self.encoder)
        self.ckpt_path = None

        

    def test_step(self, batch, batch_idx):

        device = batch["context"]["image"].device
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # get overlap.
        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        if overlap_tag == "ignore":
            return

        gt_pose = batch["context"]["extrinsics"][0, -1]
        # runing encoder to obtain the 3DGS
        visualization_dump = {}
       
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)
        
        if self.encoder.cfg.estimating_focal:
            pred_intrinsics = encoder_output['intrinsics']['c']
            context_intrinsics = pred_intrinsics
            print("estimate focal", context_intrinsics[0,0,0,0]*w, "gt focal", batch["context"]["intrinsics"][0,0,0,0]*w)
        else:
            context_intrinsics = batch["context"]["intrinsics"]
        

        assert 'means' in visualization_dump or self.encoder.cfg.estimating_pose

        all_metrics = {}
        if 'means' in visualization_dump:
            pnp_extrinsics = get_pnp_pose(visualization_dump['means'][0, -1].squeeze(),
                                        visualization_dump['opacities'][0, -1].squeeze(),
                                        context_intrinsics[0, -1], h, w,  opacity_threshold=0.2)
                
            pnp_pose = pnp_extrinsics.to(self.device)
        
            pnp_error_R, pnp_error_t = compute_pose_error_for_batch(gt_pose, pnp_pose)
            pnp_error_pose = torch.max(pnp_error_t, pnp_error_R)  # find the max error
        
            all_metrics.update({
                "pnp_e_t_ours": pnp_error_t,
                "pnp_e_R_ours": pnp_error_R,
                "pnp_e_pose_ours": pnp_error_pose,
            })

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics =  encoder_output['extrinsics']['c']
            pred_pose = pred_extrinsics[0,-1]
            error_R, error_t = compute_pose_error_for_batch(gt_pose, pred_pose)
            error_pose = torch.max(error_t, error_R)  # find the max error

            all_metrics.update({
                "e_t_ours": error_t,
                "e_R_ours": error_R,
                "e_pose_ours": error_pose
            })
        
        self.print_preview_metrics(all_metrics, overlap_tag)

        return 0

    def calculate_auc(self, tot_e_pose, method_name, overlap_tag):
        thresholds = [5, 10, 20]
        auc = pose_auc(tot_e_pose, thresholds)
        print(f"Pose AUC {method_name} {overlap_tag}: ")
        print(auc)
        return auc

    def on_test_end(self) -> None:
        # eval pose
        
       

        for item in ['R', 't', 'pose']:
            print("*"*20)
            for method in self.cfg.methods:
                if f"e_{item}_{method.key}" in self.all_mertrics:
                    tot_e_pose = np.array(self.all_mertrics[f"e_{item}_{method.key}"])
                    tot_e_pose = np.array(tot_e_pose)
                    thresholds = [5, 10, 20]
                    auc = pose_auc(tot_e_pose, thresholds)
                    self.running_metrics[f'{item}_auc'] = auc
                    
                    median_error = np.median(tot_e_pose)
                    self.running_metrics[f'{item}_median'] = median_error
                    # print(f"Pose median error {method.key} of {item}: ", median_error)
                    print(f"Pose AUC {method.key} of {item}: ", auc, " median error", median_error)

                    for overlap_tag in self.all_mertrics_sub.keys():
                        tot_e_pose = np.array(self.all_mertrics_sub[overlap_tag][f"e_{item}_{method.key}"])
                        tot_e_pose = np.array(tot_e_pose)
                        thresholds = [5, 10, 20]
                        auc = pose_auc(tot_e_pose, thresholds)
                        self.running_metrics_sub[overlap_tag][f'{item}_auc'] = auc
                        
                        median_error = np.median(tot_e_pose)
                        self.running_metrics_sub[overlap_tag][f'{item}_median'] = median_error
                        # print(f"Pose median error {method.key} {overlap_tag}  of {item}: ", median_error)
                        print(f"Pose AUC {method.key} {overlap_tag} of {item}: ", auc, " median error", median_error)


            print("*"*20)
            for method in self.cfg.methods:
                if f"pnp_e_{item}_{method.key}" in self.all_mertrics:
                    tot_e_pose = np.array(self.all_mertrics[f"pnp_e_{item}_{method.key}"])
                    tot_e_pose = np.array(tot_e_pose)
                    thresholds = [5, 10, 20]
                    auc = pose_auc(tot_e_pose, thresholds)
                    self.running_metrics[f'pnp_{item}_auc'] = auc
                    
                    median_error = np.median(tot_e_pose)
                    self.running_metrics[f'pnp_{item}_median'] = median_error
                    # print(f"PnP Pose median error {method.key} of {item}: ", median_error)
                    print(f"PnP Pose AUC {method.key} of {item}: ", auc, " median error", median_error)

                    for overlap_tag in self.all_mertrics_sub.keys():
                        tot_e_pose = np.array(self.all_mertrics_sub[overlap_tag][f"pnp_e_{item}_{method.key}"])
                        tot_e_pose = np.array(tot_e_pose)
                        thresholds = [5, 10, 20]
                        auc = pose_auc(tot_e_pose, thresholds)
                        self.running_metrics_sub[overlap_tag][f'pnp_{item}_auc'] = auc
                        
                        median_error = np.median(tot_e_pose)
                        self.running_metrics_sub[overlap_tag][f'pnp_{item}_median'] = median_error
                        # print(f"PnP Pose median error {method.key} {overlap_tag} of {item}: ", median_error)
                        print(f"PnP Pose AUC {method.key} {overlap_tag} of {item}: ", auc, " median error", median_error)

        def convert_tensors_to_values(metrics_dict):
            # return {k: convert_tensors_to_values(v) if isinstance(v, dict) else v.item() for k, v in metrics_dict.items()}
            return {k: convert_tensors_to_values(v) if isinstance(v, dict) 
                    else v.item() if isinstance(v, torch.Tensor) 
                    else v for k, v in metrics_dict.items()}
        
        
        name = get_cfg()["wandb"]["name"]
        os.makedirs(os.path.join(self.cfg.output_metrics_path,name), exist_ok=True)

        
        if self.ckpt_path is not None:
            with open(self.cfg.output_metrics_path / name  / "test_pose_ckpt_path.txt", "w") as f:
                f.write(f"{self.ckpt_path}\n")

        running_metrics = convert_tensors_to_values(self.running_metrics)
        with (self.cfg.output_metrics_path / name  / f"pose_estimation_all_avg.json").open("w") as f:
            json.dump(running_metrics, f, indent=2)

        running_metrics_sub = convert_tensors_to_values(self.running_metrics_sub)
        with (self.cfg.output_metrics_path / name  / f"pose_estimation_sub_avg.json").open("w") as f:
            json.dump(running_metrics_sub, f, indent=2)
        # # save all metrics
        np.save(self.cfg.output_metrics_path / name  / "pose_all_metrics.npy", self.all_mertrics)
        np.save(self.cfg.output_metrics_path / name  / "pose_all_metrics_sub.npy", self.all_mertrics_sub)

    def print_preview_metrics(self, metrics: dict[str, float], overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1

            self.all_mertrics = {k: [v.cpu().item()] for k, v in metrics.items()}
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

            for k, v in metrics.items():
                self.all_mertrics[k].append(v.cpu().item())

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
                self.all_mertrics_sub = {overlap_tag: {k: [v.cpu().item()] for k, v in metrics.items()}}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
                self.all_mertrics_sub[overlap_tag] = {k: [v.cpu().item()] for k, v in metrics.items()}
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

                for k, v in metrics.items():
                    self.all_mertrics_sub[overlap_tag][k].append(v.cpu().item())


        def print_metrics(runing_metric, metrics_list):
            table = []
            for method in self.cfg.methods:
                row = [
                    f"{runing_metric[f'{metric}']:.3f}"
                    for metric in metrics_list
                ]
                table.append((method.key, *row))

            table = tabulate(table, ["Method"] + metrics_list)
            print(table)

        print("All Pairs:")
        
        metrics_list = list(metrics.keys())
        print_metrics(self.running_metrics, metrics_list)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, metrics_list)
