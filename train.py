#
# Edited by: Jingwei Xu, ShanghaiTech University  
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#
# Configuration System:
# - Use --config <path> to specify custom configuration file
# - See configs/training_config.yaml for all available options
# - Use configs/simple_config.yaml as template for custom configs
# - Configuration utilities are in utils/config_utils.py
#

import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
import yaml
from datetime import datetime
from utils.loss_utils import l1_loss, ssim
from utils.config_utils import load_training_config, print_config_summary
from gaussian_renderer import render, render_semantic
import sys
from scene import Scene, GaussianModel
from scene.env_map import SkyModel
from utils.general_utils import safe_state, requires_grad, save_tensor_as_colormap_image
from utils.system_utils import mkdir_p
import uuid
from alive_progress import alive_bar
from utils.image_utils import psnr
from utils.semantic_utils import concerned_classes_ind_map, concerned_classes_list, semantic_prob_to_rgb
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.mono_priors.img_feature_extractors import get_feature_extractor, predict_img_features
from utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth, compute_metric_depth
from utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp, get_uncertainty_and_loss, get_viewpoint_uncertainty_no_grad
from utils.dyn_uncertainty.mapping_utils import compute_mapping_loss_components

try:
    import wandb
    from utils.wandb_utils import init_wandb, log_scalar, log_image, log_histogram, log_metrics, finish_wandb, is_wandb_available
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, continue_model_path, start_iteration, debug_from, config_path="configs/training_config.yaml"):
    start_time = time.time()
    
    # Load training configuration
    config = load_training_config(config_path)
    print_config_summary(config)
    first_iter = 0
    
    # Initialize W&B logging
    wandb_enabled = False
    if WANDB_AVAILABLE:
        # Create experiment name with date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = getattr(dataset, 'model_path', 'experiment').split('/')[-1]
        experiment_name = f"{base_name}_{current_time}"
        
        wandb_enabled = init_wandb(
            project_name="StreetUnveiler",
            experiment_name=experiment_name,
            config=config,
            model_path=dataset.model_path
        )
    
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    
    # Initialize uncertainty MLP if enabled
    uncertainty_mlp = None
    if hasattr(opt, 'enable_uncertainty_loss') and opt.enable_uncertainty_loss:
        uncertainty_lr = getattr(opt, 'uncertainty_lr', 0.0001)
        uncertainty_mlp = generate_uncertainty_mlp(384, lr=uncertainty_lr, setup_training=True)
        print("Uncertainty MLP initialized")
    
    if continue_model_path:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp, load_iteration=start_iteration)
    else:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp)
    gaussians.training_setup(opt)

    # Initialize depth estimator and feature extractor if needed
    feature_extractor = None
    depth_estimator = None
    feature_cfg = None
    if hasattr(opt, 'enable_feature_loss') and opt.enable_feature_loss:
        # Get feature configuration from config and update dynamic values
        feature_cfg = config['feature_cfg'].copy()
        
        # Override with command line arguments if provided
        if hasattr(opt, 'feature_extractor_model'):
            feature_cfg['mono_prior']['feature_extractor'] = opt.feature_extractor_model
        if hasattr(opt, 'depth_estimator_model'):
            feature_cfg['mono_prior']['depth'] = opt.depth_estimator_model
            
        # Set dynamic values
        feature_cfg['data']['output'] = dataset.model_path
        if hasattr(dataset, 'scene_name'):
            feature_cfg['scene'] = dataset.scene_name
        
        # Initialize depth estimator for depth-based feature loss
        depth_estimator = get_metric_depth_estimator(feature_cfg)
        print(f"Depth estimator initialized: {feature_cfg['mono_prior']['depth']}")
        
        # Initialize feature extractor only if uncertainty loss is enabled
        if hasattr(opt, 'enable_uncertainty_loss') and opt.enable_uncertainty_loss:
            feature_extractor = get_feature_extractor(feature_cfg)
            print(f"Feature extractor initialized for uncertainty: {feature_cfg['mono_prior']['feature_extractor']}")

    # may have some problems
    if continue_model_path:
        (model_params, first_iter) = torch.load(os.path.join(continue_model_path, "checkpoint", "iteration_{}".format(start_iteration), "splatting.pt"), weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    densification_cfg = config['densification']
    opt.densification_interval = int(len(scene.getTrainCameras()) * densification_cfg['densification_multiplier'])
    print(opt.densification_interval)
    ema_loss_for_log = 0.0
    first_iter += 1
    gaussians.prune_semantic_splatting(1 << concerned_classes_ind_map['sky'])
    
    # Store uncertainty for visualization
    current_uncertainty = None
    
    # Get loss weights configuration for use throughout training
    loss_weights_cfg = config['loss_weights']
    
    # Pre-compute GT depths once at the beginning (they don't change during training)
    print("Pre-computing GT depths for all training cameras...")
    computed_gt_depths = {}
    if depth_estimator is not None:
        with torch.no_grad():
            with alive_bar(len(scene.getTrainCameras()), title="Pre-computing GT depths", bar="smooth") as bar:
                for idx, viewpoint_cam in enumerate(scene.getTrainCameras()):
                    gt_image = viewpoint_cam.original_image.cuda()
                    if gt_image.dim() == 3:
                        gt_image_input = gt_image.unsqueeze(0)
                    else:
                        gt_image_input = gt_image
                    
                    # Compute GT depth once and store it
                    gt_depth = compute_metric_depth(
                        depth_estimator,
                        idx,
                        gt_image_input,
                        feature_cfg,
                        None  # No need for rendered depth for shape matching
                    )
                    computed_gt_depths[idx] = gt_depth.detach().clone()
                    bar()  # Update progress
        print(f"Pre-computed GT depths for {len(computed_gt_depths)} frames")
        
        # Store computed depths in scene for later access
        scene.computed_gt_depths = computed_gt_depths
    else:
        # Initialize empty dict if no depth estimator
        scene.computed_gt_depths = {}
    
    # # Main training loop with alive-progress
    with alive_bar(opt.iterations - first_iter + 1, title="🚀 Training Progress", bar="smooth", spinner="dots_waves") as bar:
        for iteration in range(first_iter, opt.iterations + 1):

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = [i for i in range(len(scene.getTrainCameras()))]

            select_frame_id = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            viewpoint_cam = scene.getTrainCameras()[select_frame_id]

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            loss_dict = {}

            if opt.enable_semantic_loss:
                render_pkg = render_semantic(viewpoint_cam, gaussians, pipe, background)
                render_semantics = render_pkg["render_semantics"]
                gt_semantic = viewpoint_cam.get_semantic_prob_image()
                semantic_loss = F.cross_entropy(render_semantics.unsqueeze(0), gt_semantic.unsqueeze(0), weight=torch.tensor(loss_weights_cfg['semantic_ce_weights']).cuda())

                loss_dict['semantic'] = semantic_loss
                semantic_loss = opt.semantic_loss_ratio * semantic_loss

                semantic_dist_loss = 0
                if iteration > opt.semantic_dist_from_iter:
                    for semantic_idx, semantic_name in enumerate(concerned_classes_list):
                        if semantic_name == 'sky':
                            continue
                        current_semantic_bit = (1 << semantic_idx)
                        single_semantic_render_pkg = render(viewpoint_cam, gaussians, pipe, background, semantic_filter_bit=current_semantic_bit, reverse_semantic=True)
                        single_semantic_rend_dist = single_semantic_render_pkg['rend_dist']
                        dist_scaling = 1.0
                        semantic_dist_loss += opt.lambda_dist * single_semantic_rend_dist.mean() * dist_scaling
                    loss_dict['Lsingle_semantic_distortion'] = semantic_dist_loss

                semantic_loss += semantic_dist_loss

                semantic_loss.backward()

            # Get initial render for both feature loss and uncertainty loss
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Depth-based feature loss computation
            if hasattr(opt, 'enable_feature_loss') and opt.enable_feature_loss and depth_estimator is not None:
                feature_loss_weight = getattr(opt, 'feature_loss_weight', 0.1)
                gt_image = viewpoint_cam.original_image.cuda()
                
                if gt_image.dim() == 3:
                    gt_image_input = gt_image.unsqueeze(0)  # Add batch dimension
                else:
                    gt_image_input = gt_image
                
                # Get rendered depth
                rendered_depth = render_pkg.get("surf_depth", None)
                if rendered_depth is not None:
                    # Compute metric depth using the helper function
                    metric_depth = compute_metric_depth(
                        depth_estimator,
                        select_frame_id,
                        gt_image_input,
                        feature_cfg,
                        rendered_depth
                    )
                    
                    # Compute depth L1 loss
                    feature_loss = F.l1_loss(rendered_depth.squeeze(), metric_depth) * feature_loss_weight
                    loss_dict['feature_depth'] = feature_loss
                    # Don't call backward here - will be added to main loss instead
                else:
                    print("Warning: No rendered depth available for depth-based feature loss")

        
            # Loss (using already rendered image from above)
            gt_image = viewpoint_cam.original_image.cuda()
            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            composite_image = render_image + sky_image * (1 - render_pkg["rend_alpha"])
            Ll1 = l1_loss(composite_image, gt_image)
            Lssim, ssim_map = ssim(composite_image, gt_image)

            # Initialize current_uncertainty for render loss computation
            current_uncertainty = None
            
            # Uncertainty loss computation using the formula: L_uncer = (L_SSIM + λ1 * L_uncer_D) / β_i^2 + λ2 * L_reg_V + λ3 * L_reg_U
            if scene.uncertainty_mlp is not None and hasattr(opt, 'enable_uncertainty_loss') and opt.enable_uncertainty_loss:
                # We need features for uncertainty prediction
                uncertainty_features = None
                
                if feature_extractor is not None:
                    # Extract features for uncertainty prediction
                    gt_image = viewpoint_cam.original_image.cuda()
                    if gt_image.dim() == 3:
                        gt_image_input = gt_image.unsqueeze(0)
                    else:
                        gt_image_input = gt_image
                    
                    uncertainty_features = predict_img_features(
                        feature_extractor,
                        select_frame_id,
                        gt_image_input,
                        feature_cfg,
                        "cuda",
                        save_feat=False
                    )
                else:
                    print("Warning: Uncertainty loss requires feature extractor to be enabled")
                    uncertainty_features = None
                    
                if uncertainty_features is not None:
                    # Get uncertainty loss parameters
                    lambda1 = getattr(opt, 'uncertainty_lambda1', loss_weights_cfg['uncertainty_lambda1'])
                    lambda2 = getattr(opt, 'uncertainty_lambda2', loss_weights_cfg['uncertainty_lambda2'])
                    lambda3 = getattr(opt, 'uncertainty_lambda3', loss_weights_cfg['uncertainty_lambda3'])
                    
                    # Get depth for uncertainty loss computation
                    depth_rendered = render_pkg.get("surf_depth", None)
                    uncertainty_metric_depth = None
                    
                    # For uncertainty loss, use predicted metric depth instead of ground truth depth
                    if depth_rendered is not None and depth_estimator is not None:
                        # Check if metric_depth was already computed in feature loss section
                        if 'metric_depth' in locals():
                            uncertainty_metric_depth = metric_depth
                        else:
                            # Compute metric depth for uncertainty loss using the helper function
                            gt_image = viewpoint_cam.original_image.cuda()
                            if gt_image.dim() == 3:
                                gt_image_input = gt_image.unsqueeze(0)
                            else:
                                gt_image_input = gt_image
                            
                            # Use pre-computed GT depth 
                            uncertainty_metric_depth = computed_gt_depths.get(select_frame_id, None)
                            if uncertainty_metric_depth is None:
                                # Fallback: compute if not pre-computed (shouldn't happen normally)
                                uncertainty_metric_depth = compute_metric_depth(
                                    depth_estimator,
                                    select_frame_id,
                                    gt_image_input,
                                    feature_cfg,
                                    depth_rendered
                                )
                    
                    # Predict uncertainty from features
                    target_size = gt_image.shape[-2:]  # (H, W)
                    uncertainty = scene.uncertainty_mlp(uncertainty_features, target_size=target_size)
                    
                    # Prepare data for mapping_utils uncertainty loss calculation
                    gt_image_3d = gt_image  # Already 3D (C, H, W)
                    rendered_image_3d = composite_image  # Already 3D (C, H, W)
                    
                    # Get opacity mask from render package and ensure correct shape
                    opacity_mask = render_pkg.get("opacity", torch.ones(gt_image.shape[-2:], device=gt_image.device))
                    if opacity_mask.dim() == 2:
                        opacity_mask = opacity_mask.unsqueeze(0)  # Add channel dimension (1, H, W)
                    
                    # Ensure uncertainty_metric_depth has correct shape (1, H, W)
                    if uncertainty_metric_depth.dim() == 2:
                        ref_depth = uncertainty_metric_depth.unsqueeze(0)
                    else:
                        ref_depth = uncertainty_metric_depth
                    
                    # Get rendered depth by applying metric depth estimator to rendered image
                    # This ensures same scale as GT depth
                    rendered_depth_3d = None
                    if depth_estimator is not None:
                        try:
                            # Apply metric depth estimator to rendered image to get same scale
                            rendered_image_input = rendered_image_3d.unsqueeze(0) if rendered_image_3d.dim() == 3 else rendered_image_3d
                            rendered_metric_depth = compute_metric_depth(
                                depth_estimator,
                                select_frame_id,
                                rendered_image_input,
                                feature_cfg,
                                None  # No need for shape matching since we're computing from scratch
                            )
                            if rendered_metric_depth.dim() == 2:
                                rendered_depth_3d = rendered_metric_depth.unsqueeze(0)
                            else:
                                rendered_depth_3d = rendered_metric_depth
                        except Exception as e:
                            print(f"[WARNING] Failed to compute metric depth for rendered image in training: {e}")
                            # Fallback to surf_depth
                            depth_rendered = render_pkg.get("surf_depth", None)
                            if depth_rendered is not None:
                                rendered_depth_3d = depth_rendered.unsqueeze(0) if depth_rendered.dim() == 2 else depth_rendered
                            else:
                                rendered_depth_3d = torch.zeros_like(ref_depth)
                    else:
                        # Fallback to surf_depth if metric depth estimator not available
                        depth_rendered = render_pkg.get("surf_depth", None)
                        if depth_rendered is not None:
                            rendered_depth_3d = depth_rendered.unsqueeze(0) if depth_rendered.dim() == 2 else depth_rendered
                        else:
                            rendered_depth_3d = torch.zeros_like(ref_depth)
                    
                    # Create visibility mask (all pixels visible)
                    visibility_mask = torch.ones_like(opacity_mask)
                    
                    # Get uncertainty configuration
                    uncertainty_cfg = config['uncertainty']
                    
                    # Compute uncertainty loss using mapping_utils
                    train_fraction = min(1.0, iteration / opt.iterations)  # Training progress
                    ssim_fraction = train_fraction  # Use same fraction for SSIM
                    
                    try:
                        uncertainty_loss_components = compute_mapping_loss_components(
                            gt_image_3d,
                            rendered_image_3d,
                            ref_depth,
                            rendered_depth_3d,
                            uncertainty,
                            opacity_mask,
                            train_fraction,
                            ssim_fraction,
                            uncertainty_cfg,  # Use uncertainty_cfg directly
                            visibility_mask
                        )
                        
                        # Extract uncertainty loss from components
                        uncertainty_loss_map, resized_uncertainty, rgb_l1_loss, depth_l1_loss = uncertainty_loss_components
                        uncertainty_loss = uncertainty_loss_map.mean()
                        
                        # Add regularization terms (lambda2, lambda3)
                        reg_v = torch.var(uncertainty)  # Variance regularization
                        reg_u = torch.log(torch.clamp(uncertainty, min=1e-6)).mean()  # Uncertainty regularization
                        uncertainty_loss += lambda2 * reg_v + lambda3 * reg_u
                        
                    except Exception as e:
                        print(f"Error in compute_mapping_loss_components: {e}")
                        print(f"gt_image shape: {gt_image_3d.shape}")
                        print(f"rendered_image shape: {rendered_image_3d.shape}")
                        print(f"ref_depth shape: {ref_depth.shape}")
                        print(f"rendered_depth shape: {rendered_depth_3d.shape}")
                        print(f"uncertainty shape: {uncertainty.shape}")
                        print(f"opacity_mask shape: {opacity_mask.shape}")
                        
                        # Fallback to simple uncertainty loss
                        uncertainty_loss = uncertainty.mean() * uncertainty_cfg['fallback_loss_weight']
                    
                    # Store uncertainty for visualization and render loss computation
                    current_uncertainty = uncertainty.detach()
                    

                    
                    # Train uncertainty MLP independently (as per paper)
                    loss_dict['uncertainty'] = uncertainty_loss
                    uncertainty_loss.backward()  # Uncertainty MLP만 업데이트

            # Compute render loss with uncertainty weighting if available
            # L_render = (λ5*L_color + λ6*L_depth) / β^2 + λ7*L_iso
            if current_uncertainty is not None:
                # Get render loss parameters
                lambda5 = getattr(opt, 'render_lambda5', loss_weights_cfg['render_lambda5'])  # Color loss weight
                lambda6 = getattr(opt, 'render_lambda6', loss_weights_cfg['render_lambda6'])  # Depth loss weight  
                lambda7 = getattr(opt, 'render_lambda7', loss_weights_cfg['render_lambda7'])  # Isotropic regularization weight
                
                # Resize uncertainty to match image dimensions if needed
                uncertainty_map = current_uncertainty.detach()  # DETACH to prevent gradient flow to uncertainty MLP
                if uncertainty_map.dim() == 2:
                    uncertainty_map = uncertainty_map.unsqueeze(0)
                if uncertainty_map.shape[-2:] != gt_image.shape[-2:]:
                    uncertainty_map = F.interpolate(
                        uncertainty_map.unsqueeze(0),
                        size=gt_image.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                else:
                    uncertainty_map = uncertainty_map.squeeze()
                
                # Recompute color loss for render loss (independent computational graph)
                render_Ll1 = l1_loss(composite_image, gt_image)
                render_Lssim, _ = ssim(composite_image, gt_image)
                L_color = (1.0 - opt.lambda_dssim) * render_Ll1 + opt.lambda_dssim * (1.0 - render_Lssim)
                
                # Compute L_depth if depth is available
                L_depth = torch.tensor(0.0, device=gt_image.device)
                depth_rendered = render_pkg.get("surf_depth", None)
                if depth_rendered is not None and hasattr(viewpoint_cam, 'depth') and viewpoint_cam.depth is not None:
                    depth_gt = torch.from_numpy(viewpoint_cam.depth).cuda().float()
                    if depth_rendered.shape != depth_gt.shape:
                        depth_gt = F.interpolate(
                            depth_gt.unsqueeze(0).unsqueeze(0),
                            size=depth_rendered.shape,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    L_depth = F.l1_loss(depth_rendered, depth_gt)
                
                # Compute β^2 (uncertainty squared) with small epsilon to avoid division by zero
                beta_squared = uncertainty_map.pow(2) + 1e-8
                
                # Compute render loss: L_render = (λ5*L_color + λ6*L_depth) / β^2 + λ7*L_iso (Eq. 6)
                uncertainty_weighted_loss = (lambda5 * L_color + lambda6 * L_depth) / beta_squared.mean()
                
                # Add isotropic regularization loss L_iso
                L_iso = uncertainty_map.mean()
                loss = uncertainty_weighted_loss + lambda7 * L_iso
                
                # Log individual components
                loss_dict['render_loss'] = loss
                loss_dict['L_color'] = L_color
                loss_dict['L_depth'] = L_depth
                loss_dict['L_iso'] = L_iso
                loss_dict['beta_mean'] = uncertainty_map.mean()
            else:
                # Fallback to basic loss without uncertainty weighting
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
            
            # Log basic components
            loss_dict['l1'] = Ll1
            loss_dict['ssim'] = Lssim

            lambda_normal = opt.lambda_normal if iteration > opt.normal_consist_from_iter else 0.0
            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()

            # loss
            loss += normal_loss

            loss_dict['Lnormal'] = normal_loss

            lambda_dist = opt.lambda_dist if iteration > opt.semantic_dist_from_iter else 0.0
            dist_loss = lambda_dist * (rend_dist).mean()
            loss += dist_loss
            loss_dict['Ldist'] = dist_loss

            lambda_shrink = opt.lambda_shrink if iteration > opt.shrinking_from_iter else 0.0
            shrink_loss = lambda_shrink * gaussians.get_opacity.mean()
            loss += shrink_loss
            loss_dict['Lshrink'] = shrink_loss

            loss.backward()  # Gaussian만 업데이트

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    bar.text = f"🔥 Loss: {ema_loss_for_log:.7f} | Iter: {iteration}/{opt.iterations}"
                    for _ in range(10):  # Update progress by 10 steps
                        bar()
                if iteration == opt.iterations:
                    bar.text = f"✅ Training Complete! Final Loss: {ema_loss_for_log:.7f}"

                # Log and save
                training_report(iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), sky_model, current_uncertainty, feature_extractor, feature_cfg, depth_estimator)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                if (
                        iteration < opt.prune_until_iter
                        and iteration > opt.prune_from_iter
                        and iteration % opt.prune_interval == 0
                ):
                    prune_mask = (gaussians.get_opacity < 0.5).squeeze()

                    # sky and vegetation may be transparent
                    sky_bit = 1 << concerned_classes_ind_map["sky"]
                    vegetation_bit = 1 << concerned_classes_ind_map["vegetation"]
                    dont_prune_semantic_bit = sky_bit | vegetation_bit

                    prune_mask *= ((gaussians.get_semantics_32bit & dont_prune_semantic_bit) == 0)
                    gaussians.prune_points(prune_mask)

                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    sky_model.optimizer.step()
                    sky_model.optimizer.zero_grad()
                    
                    # Update uncertainty MLP if enabled
                    scene.step_uncertainty_optimizer()

                if (iteration in checkpoint_iterations):
                    checkpoint_path = os.path.join(scene.model_path, "checkpoint", "iteration_{}".format(iteration))
                    mkdir_p(checkpoint_path)
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), os.path.join(checkpoint_path, "splatting.pt"))
                    sky_model.save(os.path.join(checkpoint_path, "sky_params.pt"))
                    
                    # Save uncertainty MLP checkpoint using Scene method
                    scene.save_checkpoint(iteration)

    # Generate uncertainty maps and depth maps for all input images at the end of training
    if scene.uncertainty_mlp is not None and feature_extractor is not None:
        print("\nGenerating uncertainty maps and depth maps for all input images...")
        uncertainty_output_dir = os.path.join(scene.model_path, "uncertainty_maps")
        depth_output_dir = os.path.join(scene.model_path, "depth_maps")
        rendered_depth_dir = os.path.join(depth_output_dir, "rendered_depth")
        gt_depth_dir = os.path.join(depth_output_dir, "gt_depth")
        
        os.makedirs(uncertainty_output_dir, exist_ok=True)
        os.makedirs(rendered_depth_dir, exist_ok=True)
        os.makedirs(gt_depth_dir, exist_ok=True)
        
        # Process all training cameras
        with torch.no_grad():
            scene.uncertainty_mlp.eval()
            with alive_bar(len(scene.getTrainCameras()), title="🎨 Generating Uncertainty & Depth Maps", bar="smooth", spinner="dots_waves") as bar:
                for idx, viewpoint_cam in enumerate(scene.getTrainCameras()):
                    try:
                        # Use ground truth image for uncertainty prediction
                        gt_image = viewpoint_cam.original_image.cuda()
                        if gt_image.dim() == 3:
                            gt_image_input = gt_image.unsqueeze(0)
                        else:
                            gt_image_input = gt_image
                        
                        # Extract features for uncertainty prediction only
                        uncertainty_features = predict_img_features(
                            feature_extractor,
                            idx,
                            gt_image_input,
                            feature_cfg,
                            "cuda",
                            save_feat=False
                        )
                        
                        # Render the scene to get rendered depth (only once)
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                        depth_rendered = render_pkg.get("surf_depth", None)
                        
                        # Use pre-computed GT depth (computed once at the beginning)
                        gt_depth = computed_gt_depths.get(idx, None)
                        
                        # Frame name for saving
                        frame_name = f"{idx:05d}"
                        if hasattr(viewpoint_cam, 'image_name'):
                            frame_name = viewpoint_cam.image_name
                        
                        from utils.general_utils import colormap
                        
                        # Save uncertainty maps
                        if uncertainty_features is not None:
                            # Store features in viewpoint temporarily
                            viewpoint_cam.features = uncertainty_features
                            
                            # Get uncertainty configuration
                            from utils.config_utils import get_default_config
                            uncertainty_cfg = get_default_config()['uncertainty']
                            
                            # Use the improved uncertainty computation function
                            uncertainty = get_viewpoint_uncertainty_no_grad(
                                scene.uncertainty_mlp, 
                                viewpoint_cam,
                                uncer_params=uncertainty_cfg,
                                device="cuda"
                            )
                            
                            # Save uncertainty visualization with jet colormap (.png)
                            uncertainty_save_path = os.path.join(uncertainty_output_dir, f"{frame_name}_uncertainty.png")
                            if not save_tensor_as_colormap_image(uncertainty, uncertainty_save_path, 'jet', frame_name):
                                continue
                        
                        # Save rendered depth map (from Gaussian splatting)
                        if depth_rendered is not None:
                            rendered_depth_save_path = os.path.join(rendered_depth_dir, f"{frame_name}_rendered_depth.png")
                            if not save_tensor_as_colormap_image(depth_rendered, rendered_depth_save_path, 'jet', frame_name):
                                continue
                        
                        # Save GT depth map (from depth estimator)
                        if gt_depth is not None:
                            gt_depth_save_path = os.path.join(gt_depth_dir, f"{frame_name}_gt_depth.png")
                            if not save_tensor_as_colormap_image(gt_depth, gt_depth_save_path, 'jet', frame_name):
                                continue
                        
                        bar.text = f"📸 Processing frame {frame_name}"
                        bar()  # Update progress
                        
                    except Exception as e:
                        print(f"Failed to generate maps for frame {idx}: {e}")
                        bar()  # Still update progress even on error
        
        print(f"Uncertainty maps saved to: {uncertainty_output_dir}")
        print(f"Rendered depth maps saved to: {rendered_depth_dir}")
        print(f"GT depth maps saved to: {gt_depth_dir}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(scene.model_path, "checkpoint", "computation_statistics.txt"), 'w', encoding='utf-8') as file:
        file.write("2DGS training {} seconds.".format(elapsed_time))
    
    # Log final training time and finish W&B run
    if is_wandb_available():
        log_scalar("training/total_time_seconds", elapsed_time)
        log_scalar("training/final_iteration", opt.iterations)
        finish_wandb()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, loss_dict, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, sky_model, current_uncertainty=None, feature_extractor=None, feature_cfg=None, depth_estimator=None):
    # Log training metrics to W&B
    if is_wandb_available():
        # Log individual loss components
        wandb_metrics = {}
        for key, value in loss_dict.items():
            wandb_metrics[f'train_loss/{key}_loss'] = value.item()
        
        # Log main metrics
        wandb_metrics.update({
            'train_loss/total_loss': loss.item(),
            'timing/iter_time': elapsed,
            'training/iteration': iteration
        })
        
        log_metrics(wandb_metrics, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        pick_frame_list = [i for i in range(2, 500, 70)]
        validation_configs = ({'name': 'test', 'cameras' : [] if len(scene.getTestCameras()) == 0 else [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in pick_frame_list]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in pick_frame_list]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    select_frame_id = pick_frame_list[idx]
                    env_image = sky_model.render_with_camera(viewpoint.image_height, viewpoint.image_width, viewpoint.K, viewpoint.c2w)
                    image = torch.clamp(render_pkg["render"] + (1.0 - render_pkg['rend_alpha']) * env_image, 0.0, 1.0)
                    disparity = torch.clamp((1.0 / render_pkg["surf_depth"]).nan_to_num(), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if is_wandb_available() and (idx < 8):
                        # Log images to W&B
                        from utils.general_utils import colormap
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/sky", env_image, step=iteration, caption=f"Sky render {viewpoint.image_name}")
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/render", image, step=iteration, caption=f"Final render {viewpoint.image_name}")
                        
                        # Add depth map visualization with colormap
                        if "surf_depth" in render_pkg:
                            depth_map = render_pkg["surf_depth"].squeeze()
                            
                            # Debug: Print depth statistics
                            if iteration % 1000 == 0 and idx == 0:  # Print only occasionally
                                print(f"[DEBUG] Rendered depth stats - Min: {depth_map.min().item():.4f}, Max: {depth_map.max().item():.4f}, Mean: {depth_map.mean().item():.4f}")
                            
                            # Apply jet colormap for better depth visualization
                            from utils.general_utils import colormap
                            
                            # Normalize depth to 0-1 range for proper colormap visualization
                            depth_normalized = depth_map.clone()
                            depth_min = depth_normalized.min()
                            depth_max = depth_normalized.max()
                            
                            if depth_max > depth_min:
                                depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
                            else:
                                depth_normalized = torch.ones_like(depth_normalized) * 0.5
                                
                            depth_colored_tensor = colormap(depth_normalized.cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                            
                            log_image(f"{config['name']}_view_{viewpoint.image_name}/rendered_depth", depth_colored_tensor, step=iteration, caption=f"Rendered depth (jet) {viewpoint.image_name}")
                        # Log normal and alpha maps
                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/rend_normal", rend_normal, step=iteration, caption=f"Rendered normal {viewpoint.image_name}")
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/surf_normal", surf_normal, step=iteration, caption=f"Surface normal {viewpoint.image_name}")
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/rend_alpha", rend_alpha, step=iteration, caption=f"Rendered alpha {viewpoint.image_name}")

                        # Apply jet colormap to disparity for better visualization
                        disparity_normalized = disparity.clone()
                        disparity_min = disparity_normalized.min()
                        disparity_max = disparity_normalized.max()
                        
                        if disparity_max > disparity_min:
                            disparity_normalized = (disparity_normalized - disparity_min) / (disparity_max - disparity_min)
                        else:
                            disparity_normalized = torch.ones_like(disparity_normalized) * 0.5
                            
                        disparity_colored_tensor = colormap(disparity_normalized.squeeze().cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/disparity_jet", disparity_colored_tensor, step=iteration, caption=f"Disparity (jet) {viewpoint.image_name}")

                        # Log rendering distribution with jet colormap
                        rend_dist = render_pkg["rend_dist"]
                        from utils.general_utils import colormap
                        
                        # Normalize rend_dist to 0-1 range for proper colormap visualization
                        rend_dist_normalized = rend_dist.clone()
                        rend_dist_min = rend_dist_normalized.min()
                        rend_dist_max = rend_dist_normalized.max()
                        
                        if rend_dist_max > rend_dist_min:
                            rend_dist_normalized = (rend_dist_normalized - rend_dist_min) / (rend_dist_max - rend_dist_min)
                        else:
                            rend_dist_normalized = torch.ones_like(rend_dist_normalized) * 0.5
                            
                        rend_dist_colored_tensor = colormap(rend_dist_normalized.squeeze().cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/rend_dist", rend_dist_colored_tensor, step=iteration, caption=f"Render distribution (jet) {viewpoint.image_name}")

                        # Log opacity map with jet colormap (in addition to rend_alpha already logged above)
                        opacity_map = render_pkg.get("opacity", None)
                        if opacity_map is not None:
                            from utils.general_utils import colormap
                            
                            # Normalize opacity to 0-1 range for proper colormap visualization
                            opacity_normalized = opacity_map.clone()
                            opacity_min = opacity_normalized.min()
                            opacity_max = opacity_normalized.max()
                            
                            if opacity_max > opacity_min:
                                opacity_normalized = (opacity_normalized - opacity_min) / (opacity_max - opacity_min)
                            else:
                                opacity_normalized = torch.ones_like(opacity_normalized) * 0.5
                                
                            opacity_colored_tensor = colormap(opacity_normalized.squeeze().cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                            log_image(f"{config['name']}_view_{viewpoint.image_name}/opacity", opacity_colored_tensor, step=iteration, caption=f"Opacity map (jet) {viewpoint.image_name}")

                        # Log semantic rendering
                        semantic_pkg = render_semantic(viewpoint, scene.gaussians, *renderArgs)
                        log_image(f"{config['name']}_view_{viewpoint.image_name}/rend_semantic", semantic_pkg['semantic_rgb'], step=iteration, caption=f"Semantic render {viewpoint.image_name}")
                        
                        # Log uncertainty heat map using improved get_viewpoint_uncertainty_no_grad function
                        if scene.uncertainty_mlp is not None and feature_extractor is not None:
                            try:
                                # First, extract and store features in viewpoint if not already available
                                if not hasattr(viewpoint, 'features') or viewpoint.features is None:
                                    # Extract features for uncertainty prediction
                                    gt_image_input = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
                                    uncertainty_features = predict_img_features(
                                        feature_extractor,
                                        idx,
                                        gt_image_input,
                                        feature_cfg,
                                        "cuda",
                                        save_feat=False
                                    )
                                    # Store features in viewpoint for consistency
                                    viewpoint.features = uncertainty_features
                                
                                # Get uncertainty configuration
                                from utils.config_utils import get_default_config
                                uncertainty_cfg = get_default_config()['uncertainty']
                                
                                # Use the improved uncertainty computation function
                                uncertainty_vis = get_viewpoint_uncertainty_no_grad(
                                    scene.uncertainty_mlp, 
                                    viewpoint,
                                    uncer_params=uncertainty_cfg,
                                    device="cuda"
                                )
                                
                                # Debug: Print uncertainty statistics
                                if iteration % 1000 == 0 and idx == 0:  # Print only occasionally
                                    print(f"[DEBUG] Improved Uncertainty stats - Min: {uncertainty_vis.min().item():.6f}, Max: {uncertainty_vis.max().item():.6f}, Mean: {uncertainty_vis.mean().item():.6f}")
                                
                                # Apply jet colormap directly (colormap function handles normalization automatically)
                                from utils.general_utils import colormap
                                uncertainty_colored_tensor = colormap(uncertainty_vis.cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                                
                                log_image(f"{config['name']}_view_{viewpoint.image_name}/uncertainty_map_improved", uncertainty_colored_tensor, step=iteration, caption=f"Improved Uncertainty map (jet) {viewpoint.image_name}")
                                
                            except Exception as e:
                                print(f"[WARNING] Failed to compute improved uncertainty: {e}")
                                # Fallback to current_uncertainty if available
                                if current_uncertainty is not None:
                                    # Resize uncertainty to match image dimensions if needed
                                    uncertainty_vis = current_uncertainty.clone()
                                    if uncertainty_vis.shape != (viewpoint.image_height, viewpoint.image_width):
                                        uncertainty_vis = F.interpolate(
                                            uncertainty_vis.unsqueeze(0).unsqueeze(0),
                                            size=(viewpoint.image_height, viewpoint.image_width),
                                            mode='bilinear',
                                            align_corners=False
                                        ).squeeze()
                                    
                                    # Apply jet colormap directly (colormap function handles normalization automatically)
                                    from utils.general_utils import colormap
                                    uncertainty_colored_tensor = colormap(uncertainty_vis.cpu().numpy(), cmap='jet')  # colormap returns CHW tensor
                                    
                                    log_image(f"{config['name']}_view_{viewpoint.image_name}/uncertainty_map_fallback", uncertainty_colored_tensor, step=iteration, caption=f"Fallback Uncertainty map (jet) {viewpoint.image_name}")
                            
                            # Log compute_mapping_loss_components debug info if feature_extractor is available
                            if feature_extractor is not None and scene.uncertainty_mlp is not None and idx < 3:  # Only for first 3 views to avoid spam
                                try:
                                    from utils.dyn_uncertainty.mapping_utils import compute_mapping_loss_components
                                    
                                    # Prepare data for compute_mapping_loss_components (same as in training)
                                    gt_image_3d = gt_image  # Already 3D (C, H, W)
                                    rendered_image_3d = image  # Final rendered image
                                    
                                    # Get rendered depth by applying metric depth estimator to rendered image
                                    # This ensures same scale as GT depth
                                    rendered_depth_3d = None
                                    if hasattr(scene, 'computed_gt_depths') and depth_estimator is not None:
                                        try:
                                            # Apply metric depth estimator to rendered image to get same scale
                                            rendered_image_input = rendered_image_3d.unsqueeze(0) if rendered_image_3d.dim() == 3 else rendered_image_3d
                                            rendered_metric_depth = compute_metric_depth(
                                                depth_estimator,
                                                select_frame_id,
                                                rendered_image_input,
                                                feature_cfg,
                                                None  # No need for shape matching since we're computing from scratch
                                            )
                                            if rendered_metric_depth.dim() == 2:
                                                rendered_depth_3d = rendered_metric_depth.unsqueeze(0)
                                            else:
                                                rendered_depth_3d = rendered_metric_depth
                                        except Exception as e:
                                            print(f"[WARNING] Failed to compute metric depth for rendered image: {e}")
                                            # Fallback to surf_depth
                                            surf_depth = render_pkg.get("surf_depth", None)
                                            if surf_depth is not None:
                                                rendered_depth_3d = surf_depth.unsqueeze(0) if surf_depth.dim() == 2 else surf_depth
                                            else:
                                                rendered_depth_3d = torch.zeros((1, viewpoint.image_height, viewpoint.image_width), device=gt_image.device)
                                    else:
                                        # Fallback to surf_depth if metric depth estimator not available
                                        surf_depth = render_pkg.get("surf_depth", None)
                                        if surf_depth is not None:
                                            rendered_depth_3d = surf_depth.unsqueeze(0) if surf_depth.dim() == 2 else surf_depth
                                        else:
                                            rendered_depth_3d = torch.zeros((1, viewpoint.image_height, viewpoint.image_width), device=gt_image.device)
                                    
                                    # Get reference depth (ground truth depth)
                                    ref_depth = None
                                    if hasattr(viewpoint, 'depth') and viewpoint.depth is not None:
                                        ref_depth = torch.from_numpy(viewpoint.depth).cuda().float()
                                        if ref_depth.dim() == 2:
                                            ref_depth = ref_depth.unsqueeze(0)
                                    else:
                                        # Use pre-computed depth if available
                                        if hasattr(scene, 'computed_gt_depths') and select_frame_id in scene.computed_gt_depths:
                                            ref_depth = scene.computed_gt_depths[select_frame_id]
                                            if ref_depth.dim() == 2:
                                                ref_depth = ref_depth.unsqueeze(0)
                                        else:
                                            ref_depth = torch.zeros_like(rendered_depth_3d)
                                    
                                    # Get opacity mask
                                    opacity_mask = render_pkg.get("rend_alpha", torch.ones(gt_image.shape[-2:], device=gt_image.device))
                                    if opacity_mask.dim() == 2:
                                        opacity_mask = opacity_mask.unsqueeze(0)
                                    
                                    # Create visibility mask
                                    visibility_mask = torch.ones_like(opacity_mask)
                                    
                                    # Get uncertainty configuration
                                    from utils.config_utils import get_default_config
                                    uncertainty_cfg = get_default_config()['uncertainty']
                                    
                                    # Call compute_mapping_loss_components with debug info
                                    train_fraction = min(1.0, iteration / 50000)  # Assume max iterations = 50000
                                    ssim_fraction = train_fraction
                                    
                                    debug_results = compute_mapping_loss_components(
                                        gt_image_3d,
                                        rendered_image_3d,
                                        ref_depth,
                                        rendered_depth_3d,
                                        uncertainty_vis,  # Use resized uncertainty
                                        opacity_mask,
                                        train_fraction,
                                        ssim_fraction,
                                        uncertainty_cfg,
                                        visibility_mask,
                                        return_debug_info=True
                                    )
                                    
                                    # Extract debug components with additional small_depth_loss components
                                    uncertainty_loss_map, resized_uncertainty, rgb_l1_loss, depth_l1_loss, depth_mask, small_ssim_loss, small_opacity, small_depth, ssim_loss, rendered_depth_masked, ref_depth_masked, small_depth_loss_before_penalize, small_depth_loss = debug_results
                                    
                                    # Log debug images to wandb with jet colormap for better visualization
                                    from utils.general_utils import colormap
                                    def apply_jet_colormap(tensor):
                                        tensor_np = tensor.squeeze().cpu().numpy()
                                        
                                        # Normalize to 0-1 range for proper colormap visualization
                                        tensor_min = tensor_np.min()
                                        tensor_max = tensor_np.max()
                                        
                                        if tensor_max > tensor_min:
                                            tensor_normalized = (tensor_np - tensor_min) / (tensor_max - tensor_min)
                                        else:
                                            tensor_normalized = np.ones_like(tensor_np) * 0.5
                                            
                                        return colormap(tensor_normalized, cmap='jet')  # colormap returns CHW tensor
                                    
                                    # Log depth mask (binary, so keep as grayscale)
                                    if depth_mask is not None:
                                        depth_mask_vis = depth_mask.squeeze() if depth_mask.dim() > 2 else depth_mask
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_depth_mask", depth_mask_vis.float(), step=iteration, caption=f"Depth mask {viewpoint.image_name}")
                                    
                                    # Log ref_depth
                                    if ref_depth is not None:
                                        ref_depth_jet = apply_jet_colormap(ref_depth)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_ref_depth", ref_depth_jet, step=iteration, caption=f"Reference depth (jet) {viewpoint.image_name}")
                                    
                                    # Log rendered_depth
                                    if rendered_depth_3d is not None:
                                        rendered_depth_jet = apply_jet_colormap(rendered_depth_3d)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_rendered_depth", rendered_depth_jet, step=iteration, caption=f"Rendered depth (jet) {viewpoint.image_name}")
                                    
                                    # Log small_ssim_loss
                                    if small_ssim_loss is not None:
                                        small_ssim_jet = apply_jet_colormap(small_ssim_loss)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_small_ssim_loss", small_ssim_jet, step=iteration, caption=f"Small SSIM loss (jet) {viewpoint.image_name}")
                                    
                                    # Log small_opacity
                                    if small_opacity is not None:
                                        small_opacity_jet = apply_jet_colormap(small_opacity)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_small_opacity", small_opacity_jet, step=iteration, caption=f"Small opacity (jet) {viewpoint.image_name}")
                                    
                                    # Log small_depth
                                    if small_depth is not None:
                                        small_depth_jet = apply_jet_colormap(small_depth)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_small_depth", small_depth_jet, step=iteration, caption=f"Small depth (jet) {viewpoint.image_name}")
                                    
                                    # Log ssim_loss
                                    if ssim_loss is not None:
                                        ssim_loss_jet = apply_jet_colormap(ssim_loss)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_ssim_loss_jet", ssim_loss_jet, step=iteration, caption=f"SSIM loss (jet) {viewpoint.image_name}")
                                    
                                    # Log rendered_depth_masked (rendered_depth * depth_mask)
                                    if rendered_depth_masked is not None:
                                        rendered_depth_masked_jet = apply_jet_colormap(rendered_depth_masked)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_rendered_depth_masked", rendered_depth_masked_jet, step=iteration, caption=f"Rendered depth masked (jet) {viewpoint.image_name}")
                                    
                                    # Log ref_depth_masked (ref_depth * depth_mask)
                                    if ref_depth_masked is not None:
                                        ref_depth_masked_jet = apply_jet_colormap(ref_depth_masked)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_ref_depth_masked", ref_depth_masked_jet, step=iteration, caption=f"Reference depth masked (jet) {viewpoint.image_name}")
                                    
                                    # Log small_depth_loss_before_penalize (penalize far away pixels 이전)
                                    if small_depth_loss_before_penalize is not None:
                                        small_depth_loss_before_jet = apply_jet_colormap(small_depth_loss_before_penalize)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_small_depth_loss_before_penalize", small_depth_loss_before_jet, step=iteration, caption=f"Small depth loss before penalize (jet) {viewpoint.image_name}")
                                    
                                    # Log small_depth_loss (penalize far away pixels 이후)
                                    if small_depth_loss is not None:
                                        small_depth_loss_after_jet = apply_jet_colormap(small_depth_loss)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_small_depth_loss_after_penalize", small_depth_loss_after_jet, step=iteration, caption=f"Small depth loss after penalize (jet) {viewpoint.image_name}")
                                        
                                except Exception as e:
                                    print(f"[WARNING] Failed to compute debug info for mapping loss components: {e}")
                        else:
                            # Debug: Print when uncertainty is None
                            if iteration % 1000 == 0 and idx == 0:
                                print(f"[DEBUG] current_uncertainty is None at iteration {iteration}")
                                print(f"[DEBUG] Uncertainty loss enabled: {hasattr(opt, 'enable_uncertainty_loss') and getattr(opt, 'enable_uncertainty_loss', False)}")
                                print(f"[DEBUG] Scene has uncertainty MLP: {scene.uncertainty_mlp is not None}")
                        
                        # Log uncertainty features if available (use viewpoint.features directly since it contains uncertainty features)
                        if hasattr(viewpoint, 'features') and viewpoint.features is not None and scene.uncertainty_mlp is not None:
                            try:
                                # Use the uncertainty features stored in viewpoint (no need to extract again)
                                uncertainty_features = viewpoint.features
                                
                                # Also test uncertainty prediction during validation
                                target_size = gt_image.shape[-2:]
                                test_uncertainty = scene.uncertainty_mlp(uncertainty_features, target_size=target_size)
                                
                                # Log uncertainty statistics for validation
                                uncertainty_stats = {
                                    f'validation_uncertainty/{config["name"]}_view_{viewpoint.image_name}_min': test_uncertainty.min().item(),
                                    f'validation_uncertainty/{config["name"]}_view_{viewpoint.image_name}_max': test_uncertainty.max().item(),
                                    f'validation_uncertainty/{config["name"]}_view_{viewpoint.image_name}_mean': test_uncertainty.mean().item(),
                                    f'validation_uncertainty/{config["name"]}_view_{viewpoint.image_name}_std': test_uncertainty.std().item(),
                                }
                                log_metrics(uncertainty_stats, step=iteration)
                                
                                if idx < 3:  # Only log first few images to avoid spam
                                    # Log uncertainty features resized to match original image dimensions
                                    target_height, target_width = gt_image.shape[-2:]
                                    
                                    # uncertainty_features is (H, W, C) format from DINO
                                    if uncertainty_features.dim() == 3:
                                        # Convert (H, W, C) to (C, H, W) for interpolation
                                        features_chw = uncertainty_features.permute(2, 0, 1)  # (C, H, W)
                                        
                                        # Resize to target image dimensions
                                        features_resized = F.interpolate(
                                            features_chw.unsqueeze(0),
                                            size=(target_height, target_width),
                                            mode='bilinear',
                                            align_corners=False
                                        ).squeeze(0)  # (C, H, W)
                                        
                                        # Calculate mean across all channels to get (H, W)
                                        mean_features = features_resized.mean(dim=0)  # (H, W)
                                        
                                        # Apply jet colormap for better visualization
                                        from utils.general_utils import colormap
                                        features_colored = colormap(mean_features.cpu().numpy(), cmap='jet')
                                        
                                        # Log as colored visualization with proper image dimensions
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/uncertainty_features_resized", features_colored, step=iteration, caption=f"Uncertainty features (resized to image dims + jet) {viewpoint.image_name}")
                                    
                                    else:
                                        print(f"[WARNING] Unexpected uncertainty_features dimension: {uncertainty_features.dim()}")
                                        
                            except Exception as e:
                                if iteration % 1000 == 0 and idx == 0:
                                    print(f"[WARNING] Failed to extract uncertainty features: {e}")

                        # Log ground truth images (only on first test iteration)
                        if iteration == testing_iterations[0]:
                            log_image(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image, step=iteration, caption=f"Ground truth {viewpoint.image_name}")
                            semantic_gt = semantic_prob_to_rgb(viewpoint.get_semantic_prob_image()) / 255.0
                            log_image(f"{config['name']}_view_{viewpoint.image_name}/semantic_gt", semantic_gt, step=iteration, caption=f"Semantic GT {viewpoint.image_name}")

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # Log validation metrics to W&B
                if is_wandb_available():
                    log_scalar(f"{config['name']}/l1_loss", l1_test.item(), step=iteration)
                    log_scalar(f"{config['name']}/psnr", psnr_test.item(), step=iteration)

        # Log scene statistics to W&B
        if is_wandb_available():
            log_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, step=iteration)
            log_scalar("scene/total_points", scene.gaussians.get_xyz.shape[0], step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(1, 50_000, 1_000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i for i in range(25_000, 50_000, 5_000)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_000])
    parser.add_argument("--continue_model_path", type=str, default = None)     # output/exp_dir/
    parser.add_argument("--start_iteration", type=int, default = None)
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to training configuration file")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.continue_model_path, args.start_iteration, args.debug_from, args.config)

    # All done
    print("\nTraining complete.")
