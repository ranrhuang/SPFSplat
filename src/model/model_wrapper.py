from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any, Dict

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import json
import numpy as np
import cv2
import os
import time

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, get_pnp_pose_batch
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections,render_cameras_es
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.color_map import apply_color_map_to_image
from ..misc.intrinsics_utils import estimate_intrinsics
from src.evaluation.metrics import compute_pose_error, compute_pose_error_for_batch
from ..misc.cam_utils import  pose_auc

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float
    min_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    training_context: bool
    freeze_pretrained: bool 
    freeze_backbone: bool
    freeze_pose_head: bool 



@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.test_step_outputs = {}

        if train_cfg.freeze_backbone: 
            self.freeze_params(freeze_keywords=['backbone'])

        if train_cfg.freeze_pretrained: 
            self.freeze_params(unfreeze_keywords=["gaussian_param_head", "pose_head", "intrinsic_encoder"])

        if self.train_cfg.freeze_pose_head:
            self.freeze_params(freeze_keywords=['pose_head'])

        self.ckpt_path = None

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined


        b, v_tgt, _, h, w = batch["target"]["image"].shape
        v_cxt = batch["context"]["image"].shape[1]


        # Run the model.
        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target=batch["target"] if self.encoder.cfg.estimating_pose else None)


        if self.encoder.cfg.estimating_pose:
            pred_extrinsics, pred_extrinsics_cwt = encoder_output['extrinsics']['c'], encoder_output['extrinsics']['cwt'] 
            target_extrinsics = pred_extrinsics_cwt[:, v_cxt:]
            context_extrinsics = pred_extrinsics_cwt[:, :v_cxt]
        else:
            target_extrinsics = batch["target"]["extrinsics"]
            context_extrinsics = batch["context"]["extrinsics"]

        
        target_intrinsics = batch["target"]["intrinsics"]
        context_intrinsics = batch["context"]["intrinsics"]
            

        
        total_loss = 0

        gaussians = encoder_output["gaussians"]
        # Determine decoder inputs
        extrinsics = target_extrinsics if not self.train_cfg.training_context else torch.cat([context_extrinsics, target_extrinsics], dim=1)
        intrinsics = target_intrinsics if not self.train_cfg.training_context else torch.cat([context_intrinsics, target_intrinsics], dim=1)
        near = batch["target"]["near"] if not self.train_cfg.training_context else torch.cat([batch["context"]["near"], batch["target"]["near"]], dim=1)
        far = batch["target"]["far"] if not self.train_cfg.training_context else torch.cat([batch["context"]["far"], batch["target"]["far"]], dim=1)
        target_gt = batch["target"]["image"] if not self.train_cfg.training_context else torch.cat([batch["context"]["image"], batch["target"]["image"]], dim=1)

        # Run decoder
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        # Compute PSNR
        psnr = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
        self.log(f"train/psnr", psnr.mean())

            
        # Compute and log loss.
        for loss_fn in self.losses:
            if loss_fn.name in ['mse', 'lpips']:
                loss = loss_fn.forward(output.color, target_gt, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss += loss

            if self.encoder.cfg.estimating_pose:
                if loss_fn.name == 'reproj':
                    if 'means' in visualization_dump:
                        pts3d = visualization_dump['means'].squeeze(-2) #  (b, v, h, w, 3)

                        c1_loss = loss_fn.forward(pts3d[:,0], pred_extrinsics_cwt[:,0], context_intrinsics[:,0], self.global_step)

                        c2_loss = 0
                        for cxt_i in range(1, v_cxt):
                            c2_loss += loss_fn.forward(pts3d[:,cxt_i], pred_extrinsics_cwt[:,cxt_i], context_intrinsics[:,cxt_i], self.global_step)
                        c2_loss = c2_loss / v_cxt

                        c2_only_loss = 0
                        for cxt_i in range(1, v_cxt):   
                            c2_only_loss += loss_fn.forward(pts3d[:,cxt_i], pred_extrinsics[:,cxt_i], context_intrinsics[:,cxt_i], self.global_step, detach_pts3d=True) # 

                        c2_only_loss = c2_only_loss / v_cxt

                        self.log(f"loss/{loss_fn.name}/context1", c1_loss)
                        self.log(f"loss/{loss_fn.name}/context2", c2_loss)

                        self.log(f"loss/{loss_fn.name}/c_only/context2", c2_only_loss)

                        loss = c1_loss + c2_loss + c2_only_loss
                        total_loss = total_loss + loss

                


        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False, normalize=True)
            distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
                                                    visualization_dump['means'][:, 0].squeeze(-2),
                                                    visualization_dump['means'][:, 1].squeeze(-2),
                                                    pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        self.log("loss/total", total_loss)
        


        if self.encoder.cfg.estimating_pose:
            # context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics[:,1:v_cxt], batch["context"]["extrinsics"][:,1:])
            context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:,v_cxt-1], batch["context"]["extrinsics"][:,v_cxt-1]) # only for the right context
            target_rot_error, target_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:,v_cxt:], batch["target"]["extrinsics"])

            self.log(f"train/context_angular_error", context_rot_error)
            self.log(f"train/context_transl_error", context_transl_error)
            self.log(f"train/target_angular_error", target_rot_error)
            self.log(f"train/target_transl_error", target_transl_error)

        

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"Epoch {self.current_epoch}; "
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"loss = {total_loss:.6f}; "
                 f"psnr = {psnr.mean().item():.6f}; "
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss
        
        

    def test_step(self, batch, batch_idx):
        v_cxt = batch["context"]["image"].shape[1]
        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1

        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")


        visualization_dump = {}

        if self.encoder.cfg.estimating_pose:
            # Render Gaussians.
            extrinsics_list = []
            rgb_list = []
            for target_view in range(v_tgt):
                # test one by one
                target_data = {
                    "image": batch["target"]["image"][:, target_view:target_view + 1],
                    "intrinsics": batch["target"]["intrinsics"][:, target_view:target_view + 1],
                    "near": batch["target"]["near"][:, target_view:target_view + 1],
                    "far": batch["target"]["far"][:, target_view:target_view + 1],
                }

                with self.benchmarker.time("encoder"):
                    encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target=target_data)
                    
                pred_extrinsics_cwt = encoder_output['extrinsics']['cwt'] 
                gaussians = encoder_output["gaussians"]

                if self.encoder.cfg.estimating_focal: 
                    pred_intrinsics_cwt = encoder_output['intrinsics']['cwt'] 
                    target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
                    print("estimate focal", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,target_view,0,0]*w)
                else:
                    target_intrinsics = target_data["intrinsics"]


                if self.test_cfg.align_pose:
                    output, updated_extrinsics = self.test_step_align(target_data, gaussians, target_intrinsics, initial_extrinsics=pred_extrinsics_cwt[:, v_cxt:])

                else:
                    with self.benchmarker.time("decoder", num_calls=1):
                        output = self.decoder.forward(
                            gaussians,
                            pred_extrinsics_cwt[:, v_cxt:],
                            target_intrinsics,
                            batch["target"]["near"][:,target_view:target_view+1],
                            batch["target"]["far"][:,target_view:target_view+1],
                            (h, w),
                        )

                extrinsics_list.append(pred_extrinsics_cwt[:, v_cxt:])    
                rgb_list.append(output.color)


            target_extrinsics =  torch.cat(extrinsics_list, dim=1) # (b, v, 4, 4)
            rgb_pred =  torch.cat(rgb_list, dim=1)[0] # (v, 3, h, w)


        else:
            # Render Gaussians.
            with self.benchmarker.time("encoder"):
                encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)
                

            target_extrinsics = batch["target"]["extrinsics"]

            if self.encoder.cfg.estimating_focal: 
                pred_intrinsics_cwt = encoder_output['intrinsics']['cwt'] 
                target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
                print("estimate focal", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,0,0,0]*w)
            else:
                target_intrinsics = batch["target"]["intrinsics"]

            gaussians = encoder_output['gaussians']

            # align the target pose
            if self.test_cfg.align_pose:
                output, updated_extrinsics = self.test_step_align(batch["target"], gaussians, target_intrinsics)
            else:
                with self.benchmarker.time("decoder", num_calls=v_tgt):
                    output = self.decoder.forward(
                        gaussians,
                        target_extrinsics,
                        target_intrinsics,
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
            rgb_pred = output.color[0] # (v, 3, h, w)



        (scene,) = batch["scene"]
        rgb_gt = batch["target"]["image"][0]
        
        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            all_metrics = {}

            all_metrics.update({
                f"lpips": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr": compute_psnr(rgb_gt, rgb_pred).mean(),
            })

            if self.encoder.cfg.estimating_pose:
                target_rot_error, target_transl_error = compute_pose_error_for_batch(target_extrinsics, batch["target"]["extrinsics"])
                context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:,1:v_cxt], batch["context"]["extrinsics"][:,1:v_cxt])
                all_metrics.update({
                    f"tgt_e_R": target_rot_error,
                    f"tgt_e_t": target_transl_error,
                    f"cxt_e_R": context_rot_error,
                    f"cxt_e_t": context_transl_error,
                })
            

            if scene not in self.test_step_outputs:
                self.test_step_outputs[scene] = [overlap_tag]
                self.test_step_outputs[scene] += [all_metrics[f'psnr'].item(), all_metrics[f'ssim'].item(),  all_metrics[f'lpips'].item()]

            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_image:
            # for index, context_gt in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
            #     save_image(context_gt, path / scene / f"context/{index:0>6}.png")

            # for index, target_gt in zip(batch["target"]["index"][0], rgb_gt):
            #     save_image(target_gt, path / scene / f"gt/{index:0>6}.png")

            for index, pred in zip(batch["target"]["index"][0], rgb_pred):
                save_image(pred, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])

            save_video(
                [a for a in rgb_pred],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = batch["context"]["image"][0]
            comparison = [
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)")
            ]
            save_image(add_border(hcat(*comparison)), path / f"{scene}.png")


    def test_step_align(self, target, gaussians, intrinsics, initial_extrinsics=None):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = target["image"].shape
        device = target["image"].device
        with torch.set_grad_enabled(True):
            if initial_extrinsics is not None:
                extrinsics = nn.Parameter(initial_extrinsics)
            else:
                extrinsics = nn.Parameter(target["extrinsics"])

            opt_params = []
            opt_params.append(
                {
                    "params": [extrinsics],
                    "lr": self.test_cfg.opt_lr,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        intrinsics,
                        target["near"],
                        target["far"],
                        (h, w)
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if loss_fn.name in ["mse", "lpips"]:
                            loss = loss_fn.forward(output.color, target["image"], gaussians, self.global_step)
                            total_loss = total_loss + loss

                    total_loss.backward()

                    if (i == 0) or (i % 50 == 0) or (i == self.test_cfg.pose_align_steps - 1):
                        print(i, total_loss.item())

                    pose_optimizer.step()

        return output, extrinsics

    

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()


        if self.ckpt_path is not None:
            with open(self.test_cfg.output_path / name  / "test_ckpt_path.txt", "w") as f:
                f.write(f"{self.ckpt_path}\n")

        with (self.test_cfg.output_path / name  / f"scores_all.json").open("w") as f:
            json.dump(self.test_step_outputs, f, indent=2)

        def convert_tensors_to_values(metrics_dict):
            return {k: convert_tensors_to_values(v) if isinstance(v, dict) else v.item() for k, v in metrics_dict.items()}

        running_metrics = convert_tensors_to_values(self.running_metrics)
        with (self.test_cfg.output_path / name  / f"scores_all_avg.json").open("w") as f:
            json.dump(running_metrics, f, indent=2)

        running_metrics_sub = convert_tensors_to_values(self.running_metrics_sub)
        with (self.test_cfg.output_path / name  / f"scores_sub_avg.json").open("w") as f:
            json.dump(running_metrics_sub, f, indent=2)

        for item in ['R', 't']:
            if self.encoder.cfg.estimating_pose:
                print("*"*20)
                for method in ['ours']:
                    tot_e_pose = np.array(self.all_mertrics[f"cxt_e_{item}"])
                    tot_e_pose = np.array(tot_e_pose)
                    thresholds = [5, 10, 20]
                    auc = pose_auc(tot_e_pose, thresholds)
                    self.running_metrics[f'{item}_auc'] = auc
                    
                    median_error = np.median(tot_e_pose)
                    self.running_metrics[f'{item}_median'] = median_error
                    print(f"Pose AUC {method} of {item}: ", auc, " median error", median_error)

                    for overlap_tag in self.all_mertrics_sub.keys():
                        tot_e_pose = np.array(self.all_mertrics_sub[overlap_tag][f"cxt_e_{item}"])
                        tot_e_pose = np.array(tot_e_pose)
                        thresholds = [5, 10, 20]
                        auc = pose_auc(tot_e_pose, thresholds)
                        self.running_metrics_sub[overlap_tag][f'{item}_auc'] = auc
                        
                        median_error = np.median(tot_e_pose)
                        self.running_metrics_sub[overlap_tag][f'{item}_median'] = median_error
                        print(f"Pose AUC {method} {overlap_tag} of {item}: ", auc, " median error", median_error)


    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
                f"target = {batch['target']['index'].tolist()}"
            )


        v_cxt = batch["context"]["image"].shape[1]
        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1

        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target=batch["target"] if self.encoder.cfg.estimating_pose else None)

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics, pred_extrinsics_cwt = encoder_output['extrinsics']['c'], encoder_output['extrinsics']['cwt'] 
            target_extrinsics = pred_extrinsics_cwt[:, v_cxt:]
            context_extrinsics = pred_extrinsics_cwt[:, :v_cxt]
        else:
            target_extrinsics = batch["target"]["extrinsics"]
            context_extrinsics = batch["context"]["extrinsics"]
            
        
        if self.encoder.cfg.estimating_focal:
            pred_intrinsics, pred_intrinsics_cwt = encoder_output['intrinsics']['c'], encoder_output['intrinsics']['cwt'] 
            target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
            context_intrinsics = pred_intrinsics_cwt[:, :v_cxt]
            print("estimate focal", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,0,0,0]*w)
        else:
            target_intrinsics = batch["target"]["intrinsics"]
            context_intrinsics = batch["context"]["intrinsics"]


        gaussians = encoder_output['gaussians']
        show_context_render = True

        # Determine decoder inputs
        extrinsics = target_extrinsics if not show_context_render else torch.cat([context_extrinsics, target_extrinsics], dim=1)
        intrinsics = target_intrinsics if not show_context_render else torch.cat([context_intrinsics, target_intrinsics], dim=1)
        near = batch["target"]["near"] if not show_context_render else torch.cat([batch["context"]["near"], batch["target"]["near"]], dim=1)
        far = batch["target"]["far"] if not show_context_render else torch.cat([batch["context"]["far"], batch["target"]["far"]], dim=1)
        target_gt = batch["target"]["image"] if not show_context_render else torch.cat([batch["context"]["image"], batch["target"]["image"]], dim=1)

        # Run decoder
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        
        # Compute validation metrics.
        rgb_gt = target_gt[0]
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        if not show_context_render:
            psnr = compute_psnr(rgb_gt, rgb_pred).mean()
            self.log(f"val/psnr", psnr)
            lpips = compute_lpips(rgb_gt, rgb_pred).mean()
            self.log(f"val/lpips", lpips)
            ssim = compute_ssim(rgb_gt, rgb_pred).mean()
            self.log(f"val/ssim", ssim)

        else:    
            psnr = compute_psnr(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
            self.log(f"val/psnr", psnr)
            lpips = compute_lpips(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
            self.log(f"val/lpips", lpips)
            ssim = compute_ssim(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
            self.log(f"val/ssim", ssim)

            # Compute validation metrics.
            psnr = compute_psnr(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
            self.log(f"val/context/psnr", psnr)
            lpips = compute_lpips(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
            self.log(f"val/context/lpips", lpips)
            ssim = compute_ssim(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
            self.log(f"val/context/ssim", ssim)

        # Construct comparison image.
        context_img = batch["context"]["image"][0]
        context_img_depth = vis_depth_map(visualization_dump["depth"][0]) # (v, h, w)

        comparison = hcat(
            add_label(vcat(*context_img), "Context"),
            add_label(vcat(*context_img_depth), "Context Depth"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), f"Prediction"),
            add_label(vcat(*depth_pred), f"Depth")
        )

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

        if self.encoder.cfg.estimating_pose:
            # context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics[:,1:v_cxt], batch["context"]["extrinsics"][:,1:])
            context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:,v_cxt-1], batch["context"]["extrinsics"][:,v_cxt-1]) # only validate the right context
            target_rot_error, target_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:,v_cxt:], batch["target"]["extrinsics"])

            self.log(f"val/context_angular_error", context_rot_error)
            self.log(f"val/context_transl_error", context_transl_error)
            self.log(f"val/target_angular_error", target_rot_error)
            self.log(f"val/target_transl_error", target_transl_error)

        

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

       
        # Run video validation step.
        self.render_video_interpolation(batch)

        
        if self.train_cfg.extended_visualization:
            self.render_video_wobble(batch)
            self.render_video_interpolation_exaggerated(batch)


    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t, context_extrinsics,  context_intrinsics):
            # origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            # origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            origin_a = context_extrinsics[:, 0, :3, 3]
            origin_b = context_extrinsics[:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                # batch["context"]["extrinsics"][:, 0],
                context_extrinsics[:,0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                # batch["context"]["intrinsics"][:, 0],
                context_intrinsics[:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

   
    
    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        # _, v, _, _ = batch["context"]["extrinsics"].shape
        

        def trajectory_fn(t, context_extrinsics,  context_intrinsics):
            _, v, _, _ = context_extrinsics.shape
            extrinsics = interpolate_extrinsics(
                context_extrinsics[0, 0],
                context_extrinsics[0, -1],
               
                t,
            )
            intrinsics = interpolate_intrinsics(
                context_intrinsics[0, 0],
                context_intrinsics[0, -1],
                
                t,
            )

            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t, context_extrinsics, context_intrinsics):
            # origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            # origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            origin_a = context_extrinsics[:, 0, :3, 3]
            origin_b = context_extrinsics[:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                context_extrinsics[0, 0],
                context_extrinsics[0, -1],
                
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                context_intrinsics[0, 0],
                context_intrinsics[0, -1],
                
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.

        _, _, _, h, w = batch["context"]["image"].shape
        _, v_cxt, _, _ = batch["context"]["extrinsics"].shape

       
        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, target=None)
        gaussians = encoder_output['gaussians']

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics = encoder_output['extrinsics']['c']
            context_extrinsics = pred_extrinsics[:, :v_cxt]
        else:
            context_extrinsics = batch["context"]["extrinsics"]

        if self.encoder.cfg.estimating_focal:
            pred_intrinsics_cwt = encoder_output['intrinsics']['c']
            context_intrinsics = pred_intrinsics_cwt[:, :v_cxt]
        else:
            context_intrinsics = batch["context"]["intrinsics"]


        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t, context_extrinsics, context_intrinsics)

    

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)


        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )

        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        
        visualizations = {
            f"video/{name}": wandb.Video(video, fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        # print("running", self.running_metrics)
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

      
        metric_list = list(metrics.keys())

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    
       
    def freeze_params(self, freeze_keywords=None, unfreeze_keywords=None):
        freeze_keywords = freeze_keywords or []
        unfreeze_keywords = unfreeze_keywords or []

        for name, param in self.named_parameters():
            if unfreeze_keywords:
                if any(kw in name for kw in unfreeze_keywords):
                    param.requires_grad = True
                    print(f"Unfreezing: {name}")
                else:
                    param.requires_grad = False
                    # print(f"Freezing: {name}")
            elif freeze_keywords:
                if any(kw in name for kw in freeze_keywords):
                    param.requires_grad = False
                    print(f"Freezing: {name}")

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name or "pose_head" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )


        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * self.optimizer_cfg.min_lr_multiplier)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Perform the backward pass
        optimizer_closure()

        nan_detected = False
        large_detected = False
        
        max_grad_norm = 5  # Define the maximum norm threshold
        max_grad_value = 0  

        # Check for NaN gradients
        for name, param in self.named_parameters():

            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    self.log("nan_gradient_detected", True, on_step=True, on_epoch=False)
                    print(f"Skipping update due to NaN gradient")
                    nan_detected = True
                    break

                param_max_grad = param.grad.abs().max().item()
                if param_max_grad > max_grad_value:
                    max_grad_value = param_max_grad

                if param.grad.abs().max() > max_grad_norm:  
                    self.log(f"large_gradient_detected in {name}", param.grad.abs().max().item(), on_step=True, on_epoch=False)
                    print(f"large gradient in {name}")
                    large_detected = True
                    break
        
        self.log("max_gradient", max_grad_value, on_step=True, on_epoch=False)

        if nan_detected or large_detected:
            optimizer.zero_grad()  # Clear gradients if skipping the step
        else:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

