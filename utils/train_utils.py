import os
import sys
import uuid
from argparse import Namespace

import numpy as np
import torch
from alive_progress import alive_bar
from PIL import Image

from gaussian_renderer import render
from utils.dyn_uncertainty.mapping_utils import compute_mapping_loss_components
from utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from utils.mono_priors.img_feature_extractors import predict_img_features
from utils.mono_priors.metric_depth_estimators import compute_metric_depth
from utils.system_utils import mkdir_p
from utils.general_utils import colormap


def _save_all_uncertainty_images(scene, gaussians, sky_model, pipe, background, dataset, depth_estimator, feature_extractor, opt, iteration):
    """
    Save uncertainty images for ALL training cameras at the final iteration of training.

    Args:
        scene: Scene object containing cameras and uncertainty MLP
        gaussians: GaussianModel object
        sky_model: SkyModel object
        pipe: Pipeline parameters
        background: Background tensor
        dataset: Dataset object
        depth_estimator: Depth estimation model
        feature_extractor: Feature extraction model
        opt: Optimization parameters
        iteration: Current iteration number
    """
    if scene.uncertainty_mlp is None:
        print(f"\n[ITER {iteration}] No uncertainty MLP found, skipping uncertainty image saving")
        return

    print(f"\n[ITER {iteration}] Saving uncertainty colormap images for ALL training cameras")
    uncertainty_colormap_folder = os.path.join(scene.model_path, "uncertainty")
    mkdir_p(uncertainty_colormap_folder)

    # Process ALL training cameras to save uncertainty
    train_cams = scene.getTrainCameras()

    with alive_bar(len(train_cams), title="💾 Saving uncertainty images", bar="smooth", spinner="waves", file=sys.stderr) as bar:
        for cam_idx, viewpoint_cam in enumerate(train_cams):
            # Render to get the necessary data
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            opacity = render_pkg["rend_alpha"]

            # Get features for uncertainty prediction
            if not hasattr(viewpoint_cam, 'features') or viewpoint_cam.features is None:
                gt_image = viewpoint_cam.original_image.cuda()
                gt_in = (gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
                viewpoint_cam.features = predict_img_features(feature_extractor, cam_idx, gt_in, dataset, save_feat=False)

            # Predict uncertainty
            beta_pred = scene.uncertainty_mlp(viewpoint_cam.features)

            # Apply the same clamping as in training
            beta_pred = torch.clamp(beta_pred, min=0.05, max=1.5)

            # Get ground truth and rendered data for mapping loss computation
            gt_image = viewpoint_cam.original_image.cuda()
            composite_image = render_pkg["render"] + sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w) * (1 - opacity)

            # Get metric depths (only if depth loss is enabled)
            if hasattr(scene, 'computed_gt_depths') and scene.computed_gt_depths:
                gt_metric_depth = scene.computed_gt_depths.get(cam_idx)
            else:
                gt_metric_depth = None

            if depth_estimator is not None:
                rendered_metric_depth = compute_metric_depth(
                    depth_estimator=depth_estimator,
                    frame_id=cam_idx,
                    image_input=composite_image,
                    feature_cfg=None,
                    rendered_depth=None,
                    viewpoint_cam=viewpoint_cam,
                    dataset=dataset
                )
            else:
                rendered_metric_depth = None

            # Get rendered features
            composite_in = composite_image.unsqueeze(0) if composite_image.dim() == 3 else composite_image
            rendered_features = predict_img_features(feature_extractor, cam_idx, composite_in, dataset, save_feat=False)

            # Get opacity mask
            opacity_mask = render_pkg.get("opacity", torch.ones(gt_image.shape[-2:], device=gt_image.device))
            if opacity_mask.dim() == 2:
                opacity_mask = opacity_mask.unsqueeze(0)

            train_frac = 0.0
            ssim_frac = 0.0

            # Compute mapping loss components to get resized_uncertainty
            uncer_loss_map, beta_resized, rgb_l1_map, depth_l1_map = compute_mapping_loss_components(
                gt_img=gt_image, rendered_img=composite_image,
                ref_depth=gt_metric_depth, rendered_depth=rendered_metric_depth,
                uncertainty=beta_pred, opacity=opacity,
                train_fraction=train_frac, ssim_fraction=ssim_frac,
                opt=opt, mask=opacity_mask,
                gt_dino_features=viewpoint_cam.features,
                rendered_dino_features=rendered_features,
                return_debug_info=False
            )

            uncertainty_np = torch.nn.functional.interpolate(
                beta_pred.view(1, 1, *beta_pred.shape).float(),
                size=gt_image.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0).detach().cpu().numpy()

            # Save uncertainty with colormap only
            uncertainty_colormap = colormap(uncertainty_np)  # This returns torch.Tensor in [C, H, W] format
            colormap_filename = f"{cam_idx:05d}.png"  # Remove _colormap suffix since it's the only format
            colormap_filepath = os.path.join(uncertainty_colormap_folder, colormap_filename)

            # Convert torch tensor to numpy and reorder dimensions for PIL
            uncertainty_colormap_np = uncertainty_colormap.cpu().numpy()  # [C, H, W]
            if uncertainty_colormap_np.shape[0] == 3:  # RGB format [3, H, W]
                uncertainty_colormap_np = uncertainty_colormap_np.transpose(1, 2, 0)  # Convert to [H, W, 3]
            uncertainty_colormap_np = (uncertainty_colormap_np * 255.0).astype(np.uint8)
            colormap_image = Image.fromarray(uncertainty_colormap_np, mode='RGB')
            colormap_image.save(colormap_filepath)

            bar.text = f"Processed {cam_idx+1}/{len(train_cams)} cameras"
            bar()

    print(f"✅ Saved {len(train_cams)} uncertainty colormap images to {uncertainty_colormap_folder}")


def _init_uncertainty_mlp(opt):
    if not opt.uncertainty_enabled:
        return None
    mlp = generate_uncertainty_mlp(
        n_features=opt.uncertainty_input_features,
        lr=opt.uncertainty_lr,
        weight_decay=opt.uncertainty_weight_decay,
        hidden_dim=opt.uncertainty_hidden_dim,
        net_depth=opt.uncertainty_net_depth,
    )
    return mlp


@torch.no_grad()
def _precompute_gt_depths(scene, depth_estimator, dataset):
    if depth_estimator is None:
        print("⚠️ depth estimator is None, skip GT depth precompute")
        return
    train_cams = scene.getTrainCameras()
    scene.computed_gt_depths = {}

    with alive_bar(len(train_cams), title="💾 Precomputing GT depths", bar="smooth", spinner="waves", file=sys.stderr) as bar:
        for idx, cam in enumerate(train_cams):
            gt = cam.original_image.cuda()
            gt_in = gt.unsqueeze(0) if gt.dim() == 3 else gt
            gt_depth = compute_metric_depth(depth_estimator, idx, gt_in, None, None, cam, dataset)
            scene.computed_gt_depths[idx] = gt_depth.detach().clone()
            bar.text = f"Processed {idx+1}/{len(train_cams)} cameras"
            bar()

    print(f"✅ precomputed GT depths: {len(scene.computed_gt_depths)}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # W&B logging is handled by wandb_utils
    return None
