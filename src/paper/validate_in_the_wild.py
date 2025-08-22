from pathlib import Path

import hydra
import torch
from einops import einsum, rearrange, repeat, pack
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate
import json
from tqdm import tqdm

from ..visualization.vis_depth import viz_depth_tensor
import os
from PIL import Image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points
    from src.misc.weight_modify import checkpoint_filter_fn
    from src.misc.intrinsics_utils import estimate_intrinsics


from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.layout import add_border, hcat, vcat
from ..misc.utils import inverse_normalize, vis_depth_map
import cv2

import torchvision.transforms as tf
from ..dataset.shims.crop_shim import apply_crop_shim, apply_crop_shim_to_views
import numpy as np


base_img_dir = Path("./examples/")
SCENES = (
    ("level9_tv", [base_img_dir/"level9_1.jpg", base_img_dir/"level9_2.jpg"], 6.0, [100], 1.8, 30),
)


device = torch.device("cuda:0")



def process_image_input(image_path_list):
    to_tensor = tf.ToTensor()

    torch_images = []
    for image_path in image_path_list:
        image = Image.open(image_path)
        torch_images.append(to_tensor(image))
    
    context_images = torch.stack(torch_images)
    h,w = context_images.shape[-2:]

    # If accurate intrinsics are known, set the intrinsics here
    intrinsics = torch.Tensor([[1, 0, 0.5],
                               [0, 1, 0.5],
                               [0, 0, 1]]).unsqueeze(0).repeat(2, 1, 1)


    example = {
                "context": {
                    "intrinsics": intrinsics.unsqueeze(0).to(device),
                    "image": context_images.unsqueeze(0).to(device),
                    "near": torch.Tensor([1.0, 1.0]).unsqueeze(0).to(device),
                    "far": torch.Tensor([100.0, 100.0]).unsqueeze(0).to(device),
                }
            }
    example["context"] = apply_crop_shim_to_views(example["context"], (256,256))
    return example

    

def render_video_generic(
    batch,
    context_extrinsics,
    context_intrinsics,
    gaussians, 
    decoder,
    video_output_path,
     rgb_output_path, depth_output_path,
    num_frames: int = 30,
    smooth: bool = True,
    loop_reverse: bool = True,
) -> None:
    # Render probabilistic estimate of scene.
    # gaussians = self.encoder(batch["context"], self.global_step)

    _, _, _, h, w = batch["context"]["image"].shape


        
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
            # print("extrinsics", context_extrinsics.shape, target_extrinsics.shape, extrinsics.shape )
            return extrinsics[None], intrinsics[None]

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    extrinsics, intrinsics = trajectory_fn(t, context_extrinsics, context_intrinsics)
    

    # TODO: Interpolate near and far planes?
    near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
    output = decoder.forward(
        gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
    )

    images = [
        vcat(rgb, depth)
        for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
    ]

    rgb_save = (output.color[0][10].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    depth_save = (vis_depth_map(output.depth[0])[10].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    # print(rgb_save.shape, depth_save.shape)
    cv2.imwrite(rgb_output_path, cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_output_path, depth_save)


    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]


    video = video.transpose(0, 2, 3, 1).astype(np.uint8)  # (num_frames, h, w, 3)

    # Video settings
    fps = 30
    height, width = video.shape[1:3]


    # Define the video writer (Codec: mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' for H.264 if needed
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame in video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    out.release()  # Save the file
    print(f"Saved video to {video_output_path}")

   


FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 8
LINE_WIDTH = 1.8
LINE_COLOR = [255, 0, 0]
POINT_DENSITY = 0.5



@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    print(cfg)
    torch.manual_seed(cfg_dict.seed)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    ckpt_weights = torch.load(cfg.checkpointing.load, map_location='cpu')
    if 'model' in ckpt_weights:
        ckpt_weights = ckpt_weights['model']
        ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
        missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
    elif 'state_dict' in ckpt_weights:
        # print(ckpt_weights)
        ckpt_weights = ckpt_weights['state_dict']
        ckpt_weights = {k[8:] if k.startswith("encoder.") else k: v for k, v in ckpt_weights.items()}
        missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)

    decoder = get_decoder(cfg.model.decoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval()
    model_wrapper = model_wrapper.to(device)


    for scene, image_path_list,  far, angles, line_width, cam_div in SCENES:

        example = process_image_input(image_path_list)


        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape


        # Generate the Gaussians.
        visualization_dump = {}
        encoder_output = model_wrapper.encoder.forward(
            example["context"], visualization_dump=visualization_dump)
        gaussians = encoder_output['gaussians']
        pred_extrinsics = encoder_output['extrinsics']['c']


        if model_wrapper.encoder.cfg.estimating_focal:
            pred_intrinsics = encoder_output['intrinsics']['c']
            context_intrinsics = pred_intrinsics[:, :2]
        else:
            context_intrinsics = example["context"]["intrinsics"]

        
        # Transform means into camera space.
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        ) # (256, 256, 1, 2, 3)
        means = homogenize_points(means) # (256, 256, 1, 2, 4)
        # w2c = example["context"]["extrinsics"].inverse()[0] # (2, 4, 4)
        w2c = pred_extrinsics.inverse()[0]
        means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3] # (256, 256, 1, 2, 3)


        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool) # (256, 256, 1, 2)

        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        # print(means[..., 2])
        mask = mask & (means[..., 2] < far * 2)
        # print(mask)


        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
            )
            return element[mask][None]

        for angle in angles:
            # Define the pose we render from.

            pose = torch.eye(4, dtype=torch.float32, device=device)
            # rotation = R.from_euler("xyz", [-15, angle - 90, 0], True).as_matrix()
            rotation = R.from_euler("xyz", [0, angle - 90, 0], True).as_matrix()
            pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
            translation = torch.eye(4, dtype=torch.float32, device=device)
            # visual balance, 0.5x pyramid/frustum volume
            translation[2, 3] = far * (0.5 ** (1 / 3))
            pose = translation @ pose

            ones = torch.ones((1,), dtype=torch.float32, device=device)
            render_args = {
                # "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
                "extrinsics": pred_extrinsics[0,:1] @ pose,
                "width": ones * far * 2,
                "height": ones * far * 2,
                "near": ones * 0,
                "far": ones * far,
                "image_shape": (1024, 1024),
                "background_color": torch.zeros(
                    (1, 3), dtype=torch.float32, device=device
                ),
                "gaussian_means": trim(gaussians.means),
                "gaussian_covariances": trim(gaussians.covariances),
                "gaussian_sh_coefficients": trim(gaussians.harmonics),
                "gaussian_opacities": trim(gaussians.opacities),
                "gaussian_rotations": trim(gaussians.rotations),
                "gaussian_scales": trim(gaussians.scales)
                # "fov_degrees": 1.5,
            }

            # Render alpha (opacity).
            dump = {}
            alpha_args = {
                **render_args,
                "gaussian_sh_coefficients": torch.ones_like(
                    render_args["gaussian_sh_coefficients"][..., :1]
                ),
                "use_sh": False,
            }
            alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

            # Render (premultiplied) color.
            color = render_cuda_orthographic(**render_args)[0]

            # Render depths. Without modifying the renderer, we can only render
            # premultiplied depth, then hackily transform it into straight alpha depth,
            # which is needed for sorting.
            depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
            depth = depth.norm(dim=-1)
            depth_args = {
                **render_args,
                "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
                "use_sh": False,
            }
            depth_premultiplied = render_cuda_orthographic(**depth_args)

            # print("depth_premultiplied", depth_premultiplied.shape)
            depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

            # Save the rendering for later depth-based alpha compositing.
            layers = [(color, alpha, depth)]

            # Figure out the intrinsics from the FOV.
            fx = 0.5 / (0.5 * dump["fov_x"]).tan()
            fy = 0.5 / (0.5 * dump["fov_y"]).tan()
            dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
            dump_intrinsics[0, 0] = fx
            dump_intrinsics[1, 1] = fy
            dump_intrinsics[:2, 2] = 0.5

            

            frustum_corners = unproject_frustum_corners(
                torch.cat(
                    (
                        pred_extrinsics[0],
                        # example["target"]["extrinsics"][0],
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        context_intrinsics[0],
                        # target_intrinsics[0],
                    ),
                    dim=0,
                ),
                torch.ones((2,), dtype=torch.float32, device=device)
                * far
                / cam_div,
            )
            camera_origins = torch.cat(
                (
                    # example["context"]["extrinsics"][0, :, :3, 3],
                    pred_extrinsics[0, :, :3, 3],
                    # example["target"]["extrinsics"][0, :, :3, 3],
                ),
                dim=0,
            )

            # Generate the 3D lines that have to be computed.
            lines = []
            for corners, origin in zip(frustum_corners, camera_origins):
                for i in range(4):
                    # corners: (4, 3), origin: (3)
                    lines.append((corners[i], corners[i - 1]))
                    lines.append((corners[i], origin))

            # Generate an alpha compositing layer for each line.
            for line_idx, (a, b) in enumerate(lines): # lines : 3 views * 8 = 24
                # Start with the point whose depth is further from the camera.
                a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
                b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
                start = a if (a_depth > b_depth).all() else b
                end = b if (a_depth > b_depth).all() else a

                # Create the alpha mask (this one is clean).
                start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
                end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
                alpha = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    (1, 1, 1),
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                if line_idx // 8 <= 1: # one view = 4 corners * 2 lines = 8, this selects the first two views
                    lcolor = [255, 0, 0]
                else:
                    lcolor = [0, 255, 0]

                # Create the color.
                lc = torch.tensor(
                    lcolor,
                    dtype=torch.float32,
                    device=device,
                )
                color = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    lc,
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Create the depth. We just individually render points.
                wh = torch.tensor((w, h), dtype=torch.float32, device=device)
                delta = (wh * (start_2d - end_2d)).norm()
                num_points = delta / POINT_DENSITY
                t = torch.linspace(0, 1, int(num_points) + 1, device=device)
                xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
                depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
                depth = repeat(depth, "p -> p c", c=3)
                xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
                depth = draw_points(
                    torch.ones_like(color) * 1e10,
                    xy,
                    depth,
                    LINE_WIDTH,  # makes it 2x as wide as line
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                layers.append((color, alpha, depth))

            # Do the alpha compositing.
            canvas = torch.ones_like(color)
            colors = torch.stack([x for x, _, _ in layers]) # (8*views+1, 3, 1024, 1024)
            alphas = torch.stack([x for _, x, _ in layers]) # (8*views+1, 3, 1024, 1024)
            depths = torch.stack([x for _, _, x in layers]) # (8*views+1, 3, 1024, 1024)
            index = depths.argsort(dim=0)
            colors = colors.gather(index=index, dim=0)
            alphas = alphas.gather(index=index, dim=0)
            t = (1 - alphas).cumprod(dim=0)
            t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
            image = (t * colors).sum(dim=0)
            total_alpha = (t * alphas).sum(dim=0)
            image = total_alpha * image + (1 - total_alpha) * canvas

            image = colors.sum(0)

            base = Path(f"point_clouds/{cfg.wandb['name']}/{scene}/")
            save_image(image, base / f"{cfg.wandb['name']}_{scene}_angle_{angle:0>3}.png")

            # also save the premultiplied color for debugging
            save_image(layers[0][0], base / f"{cfg.wandb['name']}_{scene}_angle_{angle:0>3}_raw.png")

           
            # convert the rotations from camera space to world space as required
            cam_rotations = trim(visualization_dump["rotations"])[0]

            c2w_mat = repeat(
                # example["context"]["extrinsics"][0, :, :3, :3],
                pred_extrinsics[0, :, :3, :3],
                "v a b -> h w spp v a b",
                h=256,
                w=256,
                spp=1,
            )
            c2w_mat = c2w_mat[mask]  # apply trim

            cam_rotations_np = R.from_quat(
                cam_rotations.detach().cpu().numpy()
            ).as_matrix()
            world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
            world_rotations = R.from_matrix(world_mat).as_quat()
            world_rotations = torch.from_numpy(world_rotations).to(
                visualization_dump["scales"]
            )

            export_ply(
                # example["context"]["extrinsics"][0, 0],
                pred_extrinsics[0, 0],
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                world_rotations,
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                base / f"{cfg.wandb['name']}_{scene}_gaussians.ply",
            )

             # Render depth.
            *_, h, w = example["context"]["image"].shape

            video_output_path = base / f"{cfg.wandb['name']}_{scene}_video.mp4"
            rgb_output_path = base / f"{cfg.wandb['name']}_{scene}_rgb.jpg"
            depth_output_path = base / f"{cfg.wandb['name']}_{scene}_depth.jpg"
            render_video_generic(example, pred_extrinsics, context_intrinsics, gaussians, model_wrapper.decoder, video_output_path, rgb_output_path, depth_output_path)
        
            

            # save encoder depth map
            depth_vis = (
                (visualization_dump["depth"].squeeze(-1).squeeze(-1)).cpu().detach()
            )
            for v_idx in range(depth_vis.shape[1]):
                vis_depth = viz_depth_tensor(
                    1.0 / depth_vis[0, v_idx], return_numpy=True
                )  # inverse depth
                Image.fromarray(vis_depth).save(base / f"{cfg.wandb['name']}_{scene}_depth_{v_idx}.png")

            # save context views
            save_image(example["context"]["image"][0, 0], base / f"{cfg.wandb['name']}_{scene}_input_0.png")
            save_image(example["context"]["image"][0, 1], base / f"{cfg.wandb['name']}_{scene}_input_1.png")
           
            a = 1
        a = 1
    a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
