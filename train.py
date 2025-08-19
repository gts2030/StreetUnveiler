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
from utils.loss_utils import l1_loss, ssim, weighted_l1, weighted_mean_map
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
from utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp, get_viewpoint_uncertainty_no_grad
from utils.dyn_uncertainty.mapping_utils import compute_mapping_loss_components

try:
    import wandb
    from utils.wandb_utils import init_wandb, log_scalar, log_image, log_histogram, log_metrics, finish_wandb, is_wandb_available
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def precompute_gt_depths(scene, depth_estimator, feature_cfg):
    """
    Pre-compute GT depths for all training cameras and store them in the scene.
    
    Args:
        scene: Scene object containing training cameras
        depth_estimator: Depth estimation model
        feature_cfg: Feature configuration for depth estimation
    
    Raises:
        RuntimeError: If any critical error occurs during GT depth computation
    """
    print("Pre-computing GT depths for all training cameras...")
    
    # Validation checks
    if scene is None:
        raise RuntimeError("❌ Scene object is None - cannot proceed with GT depth computation")
    
    try:
        train_cameras = scene.getTrainCameras()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to get training cameras from scene: {e}")
    
    if not train_cameras or len(train_cameras) == 0:
        raise RuntimeError("❌ No training cameras found in scene - cannot compute GT depths")
    
    print(f"Found {len(train_cameras)} training cameras for GT depth computation")
    computed_gt_depths = {}
    
    if depth_estimator is not None:
        if feature_cfg is None:
            raise RuntimeError("❌ Feature configuration is None but depth estimator is provided")
        
        try:
            with torch.no_grad():
                with alive_bar(len(train_cameras), title="Pre-computing GT depths", bar="smooth") as bar:
                    for idx, viewpoint_cam in enumerate(train_cameras):
                        try:
                            # Validate viewpoint camera
                            if viewpoint_cam is None:
                                raise RuntimeError(f"❌ Viewpoint camera at index {idx} is None")
                            
                            if not hasattr(viewpoint_cam, 'original_image') or viewpoint_cam.original_image is None:
                                raise RuntimeError(f"❌ No original_image found for camera {idx}")
                            
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
                            
                            # Validate computed depth
                            if gt_depth is None:
                                raise RuntimeError(f"❌ Failed to compute GT depth for camera {idx} - result is None")
                            
                            if torch.isnan(gt_depth).any() or torch.isinf(gt_depth).any():
                                raise RuntimeError(f"❌ Invalid GT depth computed for camera {idx} - contains NaN or Inf values")
                            
                            computed_gt_depths[idx] = gt_depth.detach().clone()
                            bar()  # Update progress
                            
                        except Exception as e:
                            raise RuntimeError(f"❌ Failed to compute GT depth for camera {idx}: {e}")
        
        except RuntimeError:
            raise  # Re-raise RuntimeErrors as they are already formatted
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error during GT depth computation: {e}")
        
        # Final validation
        if len(computed_gt_depths) != len(train_cameras):
            raise RuntimeError(f"❌ GT depth computation incomplete: computed {len(computed_gt_depths)} depths for {len(train_cameras)} cameras")
        
        print(f"✅ Successfully pre-computed GT depths for {len(computed_gt_depths)} frames")
        
        # Store computed depths in scene for later access
        scene.computed_gt_depths = computed_gt_depths
        print("✅ GT depths stored in scene object")
        
    else:
        print("⚠️  Depth estimator not provided - skipping GT depth computation")
        # Keep the empty dictionary that was initialized in Scene.__init__


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, continue_model_path, start_iteration, debug_from, config_path="configs/training_config.yaml"):
    start_time = time.time()
    
    # Load training configuration
    config = load_training_config(config_path)
    print_config_summary(config)
    first_iter = 0
    
    # Initialize W&B logging
    wandb_enabled = prepare_output_and_wandb(dataset, config)
    if not wandb_enabled:
        print("W&B logging failed to initialize. Continuing without logging.")

    # Initialize Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    
    # Initialize uncertainty MLP if enabled
    uncertainty_mlp = initialize_uncertainty_mlp(config)
    
    if continue_model_path:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp, load_iteration=start_iteration)
    else:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp)
    gaussians.training_setup(opt)

    # Initialize depth estimator and feature extractor if needed
    feature_extractor, depth_estimator, feature_cfg = initialize_feature_extractors(config, dataset)
    
    # Debug: Check if feature extractor was successfully initialized
    print(f"🔍 Feature extractor after initialization: {feature_extractor is not None}")
    print(f"🔍 Depth estimator after initialization: {depth_estimator is not None}")
    print(f"🔍 Feature config after initialization: {feature_cfg is not None}")

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
    print("Densification interval: ", opt.densification_interval)
    ema_loss_for_log = 0.0
    first_iter += 1
    
    # Remove sky points
    gaussians.prune_semantic_splatting(1 << concerned_classes_ind_map['sky'])
    
    # Store uncertainty for visualization
    current_uncertainty = None
    
    # Get loss weights configuration for use throughout training
    loss_weights_cfg = config['loss_weights']
    
    # Pre-compute GT depths once at the beginning (they don't change during training)
    try:
        precompute_gt_depths(scene, depth_estimator, feature_cfg)
    except RuntimeError as e:
        print(f"\n{e}")
        print("🛑 Training cannot proceed without valid GT depths. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during GT depth pre-computation: {e}")
        print("🛑 Training cannot proceed. Exiting...")
        sys.exit(1)
    
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
            # 항상 semantic 패키지를 먼저 얻어 불확실도 맵을 확보
            semantic_pkg = render_semantic(viewpoint_cam, gaussians, pipe, background)
            render_semantics = semantic_pkg["render_semantics"]        # [6,H,W]
            sem_uncertainty = semantic_pkg["semantic_uncertainty"]     # [1,H,W]
            if opt.enable_semantic_loss:
                gt_semantic = viewpoint_cam.get_semantic_prob_image()
                semantic_loss = F.cross_entropy(render_semantics.unsqueeze(0), gt_semantic.unsqueeze(0),
                                                weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.2, 1.0]).cuda())
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

            # # Depth-based feature loss computation
            # if config['feature_cfg'].get('enabled', False) and depth_estimator is not None:
            #     feature_loss_weight = loss_weights_cfg.get('feature_loss_weight', 0.1)
            #     gt_image = viewpoint_cam.original_image.cuda()
                
            #     if gt_image.dim() == 3:
            #         gt_image_input = gt_image.unsqueeze(0)  # Add batch dimension
            #     else:
            #         gt_image_input = gt_image
                
            #     # Get rendered depth
            #     rendered_depth = render_pkg.get("surf_depth", None)
            #     if rendered_depth is not None:
            #         # Compute metric depth using the helper function
            #         metric_depth = compute_metric_depth(
            #             depth_estimator,
            #             select_frame_id,
            #             gt_image_input,
            #             feature_cfg,
            #             rendered_depth
            #         )
                    
            #         # Compute depth L1 loss
            #         feature_loss = F.l1_loss(rendered_depth.squeeze(), metric_depth) * feature_loss_weight
            #         loss_dict['feature_depth'] = feature_loss
            #         # Don't call backward here - will be added to main loss instead
            #     else:
            #         print("Warning: No rendered depth available for depth-based feature loss")

        
            # Loss (using already rendered image from above)
            gt_image = viewpoint_cam.original_image.cuda()
            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            composite_image = render_image + sky_image * (1 - render_pkg["rend_alpha"])
            Ll1 = l1_loss(composite_image, gt_image)
            Lssim, ssim_map = ssim(composite_image, gt_image)

            # Initialize current_uncertainty for render loss computation
            current_uncertainty = None
            
            # Uncertainty loss computation using the formula: L_uncer = (L_SSIM + λ1 * L_uncer_D) / β_i^2 + λ2 * L_reg_V + λ3 * L_reg_U
            if scene.uncertainty_mlp is not None and config['uncertainty'].get('enabled', False):
                # We need features for uncertainty prediction
                uncertainty_features = None
                
                # gt image의 dino features 추출
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
                    # Get pre-computed GT depth
                    gt_metric_depth = scene.computed_gt_depths.get(select_frame_id, None)
                    
                    # Predict uncertainty from gt image dino features
                    target_size = gt_image.shape[-2:]  # (H, W)
                    uncertainty = scene.uncertainty_mlp(uncertainty_features, target_size=target_size)
                    
                    # Get opacity mask from render package and ensure correct shape
                    opacity_mask = render_pkg.get("opacity", torch.ones(gt_image.shape[-2:], device=gt_image.device))
                    if opacity_mask.dim() == 2:
                        opacity_mask = opacity_mask.unsqueeze(0)  # Add channel dimension (1, H, W)
                    
                    # Ensure gt_metric_depth has correct shape (1, H, W)
                    if gt_metric_depth.dim() == 2:
                        ref_depth = gt_metric_depth.unsqueeze(0)
                    else:
                        ref_depth = gt_metric_depth
                    
                    # Get rendered depth by applying metric depth estimator to rendered image
                    # This ensures same scale as GT depth
                    rendered_depth = None
                    if depth_estimator is not None:
                        try:
                            # Apply metric depth estimator to rendered image to get same scale
                            rendered_image_input = composite_image.unsqueeze(0) if composite_image.dim() == 3 else composite_image
                            rendered_metric_depth = compute_metric_depth(
                                depth_estimator,
                                select_frame_id,
                                rendered_image_input,
                                feature_cfg,
                                None  # No need for shape matching since we're computing from scratch
                            )
                            if rendered_metric_depth.dim() == 2:
                                rendered_depth = rendered_metric_depth.unsqueeze(0)
                            else:
                                rendered_depth = rendered_metric_depth
                        except Exception as e:
                            print(f"[WARNING] Failed to compute metric depth for rendered image in training: {e}")
                            # Fallback to surf_depth
                            depth_rendered = render_pkg.get("surf_depth", None)
                            if depth_rendered is not None:
                                rendered_depth = depth_rendered.unsqueeze(0) if depth_rendered.dim() == 2 else depth_rendered
                            else:
                                rendered_depth = torch.zeros_like(ref_depth)
                    else:
                        # Fallback to surf_depth if metric depth estimator not available
                        depth_rendered = render_pkg.get("surf_depth", None)
                        if depth_rendered is not None:
                            rendered_depth = depth_rendered.unsqueeze(0) if depth_rendered.dim() == 2 else depth_rendered
                        else:
                            rendered_depth = torch.zeros_like(ref_depth)
                    
                    # Get uncertainty configuration
                    uncertainty_cfg = config['uncertainty']
                    
                    # Compute uncertainty loss using mapping_utils (following WildGS SLAM approach)
                    train_frac = uncertainty_cfg.get('train_frac_fix', 0.3)
                    ssim_frac = train_frac  # Use same fraction for SSIM
                    
                    # Apply exposure compensation if enabled (from WildGS SLAM)
                    # initialization = iteration < opt.densify_from_iter  # Consider early iterations as initialization
                    # exposure_compensated_image = composite_image
                    # if not initialization and hasattr(viewpoint_cam, 'exposure_a') and hasattr(viewpoint_cam, 'exposure_b'):
                    #     exposure_compensated_image = torch.exp(viewpoint_cam.exposure_a) * composite_image + viewpoint_cam.exposure_b
                    
                    # Create valid pixel mask (RGB boundary threshold)
                    rgb_boundary_threshold = uncertainty_cfg.get('rgb_boundary_threshold', 0.01)
                    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).unsqueeze(0)  # (1, H, W)
                    
                    try:
                        # Extract DINO features for rendered image
                        rendered_dino_features = None
                        if feature_extractor is not None:
                            try:
                                rendered_dino_features = predict_img_features(
                                    feature_extractor,
                                    select_frame_id,
                                    composite_image.unsqueeze(0),  # rendered image
                                    feature_cfg,
                                    "cuda",
                                    save_feat=False
                                )
                            except Exception as e:
                                print(f"Warning: Failed to extract DINO features from rendered image: {e}")
                        
                        uncertainty_loss_components = compute_mapping_loss_components(
                            gt_image,
                            composite_image,  # Use exposure compensated image
                            ref_depth,
                            rendered_depth,
                            uncertainty,
                            opacity_mask,
                            train_frac,  # Use fixed training fraction from config
                            ssim_frac,
                            uncertainty_cfg,
                            rgb_pixel_mask,  # Use RGB pixel mask instead of visibility_mask
                            uncertainty_features,  # GT DINO features
                            rendered_dino_features  # Rendered DINO features
                        )
                        
                        # Extract uncertainty loss from components
                        uncertainty_loss_map, resized_uncertainty, rgb_l1_loss, depth_l1_loss = uncertainty_loss_components
                        uncertainty_loss = uncertainty_loss_map.mean()
                        
                        # Add DINO feature regularization (from WildGS SLAM)
                        reg_stride = uncertainty_cfg.get('reg_stride', 4)
                        reg_mult = uncertainty_cfg.get('reg_mult', 0.1)
                        if hasattr(viewpoint_cam, 'features') and viewpoint_cam.features is not None:
                            try:
                                # Import here to avoid circular imports
                                from utils.dyn_uncertainty.mapping_utils import compute_dino_regularization_loss
                                
                                # Ensure features and uncertainty have the same spatial dimensions before stride sampling
                                features = viewpoint_cam.features.to(device=uncertainty.device)  # (H_f, W_f, C)
                                uncertainty_for_reg = resized_uncertainty  # (H_u, W_u)
                                
                                # Get target dimensions (use smaller dimensions for memory efficiency)
                                target_h = min(features.shape[0], uncertainty_for_reg.shape[0])
                                target_w = min(features.shape[1], uncertainty_for_reg.shape[1])
                                
                                # Resize both to same dimensions if needed
                                if features.shape[:2] != (target_h, target_w):
                                    # Resize features from (H_f, W_f, C) to (target_h, target_w, C)
                                    features_resized = F.interpolate(
                                        features.permute(2, 0, 1).unsqueeze(0),  # (1, C, H_f, W_f)
                                        size=(target_h, target_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0).permute(1, 2, 0)  # (target_h, target_w, C)
                                else:
                                    features_resized = features
                                
                                if uncertainty_for_reg.shape != (target_h, target_w):
                                    # Resize uncertainty from (H_u, W_u) to (target_h, target_w)
                                    uncertainty_resized = F.interpolate(
                                        uncertainty_for_reg.unsqueeze(0).unsqueeze(0),  # (1, 1, H_u, W_u)
                                        size=(target_h, target_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0).squeeze(0)  # (target_h, target_w)
                                else:
                                    uncertainty_resized = uncertainty_for_reg
                                
                                # Now apply stride sampling to tensors with matching spatial dimensions
                                feature_buffer = [
                                    features_resized[::reg_stride, ::reg_stride],  # (H_s, W_s, C)
                                ]
                                uncer_buffer = [
                                    uncertainty_resized[::reg_stride, ::reg_stride].unsqueeze(-1),  # (H_s, W_s, 1)
                                ]
                                
                                # Verify shapes match before calling regularization
                                assert feature_buffer[0].shape[:2] == uncer_buffer[0].shape[:2], \
                                    f"Shape mismatch: features {feature_buffer[0].shape[:2]} vs uncertainty {uncer_buffer[0].shape[:2]}"
                                
                                dino_reg_loss = compute_dino_regularization_loss(uncer_buffer, feature_buffer)
                                uncertainty_loss += reg_mult * dino_reg_loss
                                loss_dict['dino_reg'] = dino_reg_loss
                                
                            except Exception as e:
                                if iteration % 5000 == 0:  # Print detailed debug info occasionally
                                    print(f"[WARNING] DINO regularization failed: {e}")
                                    print(f"[DEBUG] Features shape: {getattr(viewpoint_cam, 'features', torch.empty(0)).shape}")
                                    print(f"[DEBUG] Resized uncertainty shape: {resized_uncertainty.shape}")
                                    print(f"[DEBUG] reg_stride: {reg_stride}")

                        # # ---------- Semantic-aware Regularization (intra-class consistency) ----------
                        # lambda_sem_cons = loss_weights_cfg.get('lambda_sem_cons', 0.0)
                        # if lambda_sem_cons > 0.0:
                        #     try:
                        #         # Retrieve per-pixel semantic label (H, W)
                        #         gt_semantic = viewpoint_cam.get_semantic_prob_image().to(resized_uncertainty.device)
                                
                        #         # Apply semantic consistency to multiple classes, not just car
                        #         sem_cons_total = 0.0
                        #         sem_cons_count = 0
                                
                        #         for class_name in ['car', 'building', 'road']:  # Apply to multiple semantic classes
                        #             class_idx = concerned_classes_ind_map.get(class_name, None)
                        #             if class_idx is not None:
                        #                 mask_class = (gt_semantic == class_idx)
                        #                 if mask_class.any():
                        #                     beta_class = resized_uncertainty.squeeze()[mask_class]
                        #                     if beta_class.numel() > 1:  # Need at least 2 pixels for variance
                        #                         class_mean = beta_class.mean()
                        #                         class_cons_loss = ((beta_class - class_mean) ** 2).mean()
                        #                         sem_cons_total += class_cons_loss
                        #                         sem_cons_count += 1
                        #                         loss_dict[f'sem_cons_{class_name}'] = class_cons_loss
                                
                        #         if sem_cons_count > 0:
                        #             sem_cons_loss = sem_cons_total / sem_cons_count  # Average across classes
                        #             uncertainty_loss = uncertainty_loss + lambda_sem_cons * sem_cons_loss
                        #             loss_dict['uncer_sem_cons'] = sem_cons_loss
                                    
                        #     except Exception as e:
                        #         if iteration % 5000 == 0:  # Reduce logging frequency
                        #             print(f"[WARNING] Semantic-aware regularization failed: {e}")
                        # # ---------------------------------------------------------------------------
                        
                    except Exception as e:
                        if iteration % 5000 == 0:  # Only log detailed errors every 5000 iterations
                            print(f"Error in compute_mapping_loss_components: {e}")
                            print(f"gt_image shape: {gt_image.shape}")
                            print(f"rendered_image shape: {composite_image.shape}")
                            print(f"ref_depth shape: {ref_depth.shape}")
                            print(f"rendered_depth shape: {rendered_depth.shape}")
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
            beta_squared = current_uncertainty.pow(2) + 1e-8
            if current_uncertainty is not None:
                # Get render loss parameters from config
                lambda5 = loss_weights_cfg.get('render_lambda5', 0.5)  # Color loss weight
                lambda6 = loss_weights_cfg.get('render_lambda6', 0.5)  # Depth loss weight  
                lambda7 = loss_weights_cfg.get('render_lambda7', 0.01)  # Isotropic regularization weight
                
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
                if opt.use_uncertainty_weighting:
                    # w = 1/(eps + u^power), clip으로 폭주 방지
                    w = 1.0 / (opt.uncertainty_eps + sem_uncertainty.pow(opt.uncertainty_power))
                    if opt.uncertainty_weight_clip > 0:
                        w = w.clamp(max=opt.uncertainty_weight_clip)
                    render_Ll1 = weighted_l1(composite_image, gt_image, w)
                else:
                    w = None
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
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]  # [1,H,W]
            if opt.use_uncertainty_weighting and lambda_normal > 0:
                normal_loss = lambda_normal * weighted_mean_map(normal_error, w)
            else:
                normal_loss = lambda_normal * (normal_error).mean()

            # loss
            loss += normal_loss / beta_squared.mean()

            loss_dict['Lnormal'] = normal_loss

            lambda_dist = opt.lambda_dist if iteration > opt.semantic_dist_from_iter else 0.0
            if opt.use_uncertainty_weighting and lambda_dist > 0:
                dist_loss = lambda_dist * weighted_mean_map(rend_dist, w)
            else:
                dist_loss = lambda_dist * (rend_dist).mean()
            loss += dist_loss
            loss_dict['Ldist'] = dist_loss / beta_squared.mean()

            lambda_shrink = opt.lambda_shrink if iteration > opt.shrinking_from_iter else 0.0
            shrink_loss = lambda_shrink * gaussians.get_opacity.mean()
            loss += shrink_loss
            loss_dict['Lshrink'] = shrink_loss / beta_squared.mean()
            # ----- New: 차량에 대한 불확실도 가중 shrink 정규화 -----
            if iteration > opt.vehicle_shrink_from_iter and opt.lambda_vehicle_shrink > 0:
                veh_idx = concerned_classes_ind_map['vehicle']
                veh_prob = render_semantics[veh_idx:veh_idx+1]  # [1,H,W]
                vehicle_unc_scalar = (veh_prob * sem_uncertainty).mean()
                vehicle_mask = gaussians.get_semantic_index_splatting_mask(veh_idx)
                if vehicle_mask.any():
                    veh_opacity_mean = gaussians.get_opacity[vehicle_mask].mean()
                    vehicle_shrink = opt.lambda_vehicle_shrink * vehicle_unc_scalar * veh_opacity_mean
                    loss += vehicle_shrink
                    loss_dict['Lvehicle_shrink'] = vehicle_shrink

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
                    # 기본 프루닝: 설정에서 제공하는 임계치 사용
                    prune_mask = (gaussians.get_opacity.squeeze() < opt.prune_opacity)

                    # sky and vegetation may be transparent
                    sky_bit = 1 << concerned_classes_ind_map["sky"]
                    vegetation_bit = 1 << concerned_classes_ind_map["vegetation"]
                    dont_prune_semantic_bit = sky_bit | vegetation_bit

                    prune_mask *= ((gaussians.get_semantics_32bit & dont_prune_semantic_bit) == 0)
                    # 차량은 더 공격적인 임계치로 프루닝
                    veh_mask = gaussians.get_semantic_index_splatting_mask(concerned_classes_ind_map["vehicle"])
                    prune_mask = torch.logical_or(prune_mask, torch.logical_and(veh_mask, gaussians.get_opacity.squeeze() < opt.prune_vehicle_opacity))
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
                    
                # ----- New: 주기적 차량 opacity 감쇠(불확실도 스케일 적용) -----
                if (opt.use_uncertainty_weighting and
                    iteration >= opt.uncertainty_decay_from_iter and
                    iteration % opt.uncertainty_decay_interval == 0):
                    veh_idx = concerned_classes_ind_map['vehicle']
                    veh_mask = gaussians.get_semantic_index_splatting_mask(veh_idx)
                    if veh_mask.any():
                        # 현재 뷰에서 차량 영역의 평균 불확실도를 이용해 decay 강도를 조절
                        veh_prob = render_semantics[veh_idx:veh_idx+1]
                        global_unc = (veh_prob * sem_uncertainty).mean().item()
                        # 1.0에 가까울수록 약한 감쇠, 불확실도가 높을수록 더 감쇠
                        decay = 1.0 - (1.0 - opt.uncertainty_decay_factor) * float(global_unc)
                        decay = max(0.0, min(decay, 1.0))
                        if decay < 0.999:
                            gaussians.decay_opacity_with_mask(veh_mask, decay)
                    
                

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
                        gt_depth = scene.computed_gt_depths.get(idx, None)
                        
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

def initialize_uncertainty_mlp(config):
    """
    Initialize uncertainty MLP based on configuration settings.
    
    Args:
        config: Training configuration dictionary containing uncertainty settings
    
    Returns:
        MLPNetwork or None: Initialized uncertainty MLP if enabled, None otherwise
    """
    uncertainty_cfg = config['uncertainty']
    
    # Check if uncertainty is enabled
    if not uncertainty_cfg.get('enabled', False):
        print("⚠️  Uncertainty estimation disabled in configuration")
        return None
    
    try:
        # Get uncertainty parameters from config
        uncertainty_lr = uncertainty_cfg.get('lr', 0.0004)
        uncertainty_weight_decay = uncertainty_cfg.get('weight_decay', 0.00001)
        input_features = uncertainty_cfg.get('input_features', 384)
        hidden_dim = uncertainty_cfg.get('hidden_dim', 64)
        net_depth = uncertainty_cfg.get('net_depth', 2)
        
        # Initialize uncertainty MLP
        uncertainty_mlp = generate_uncertainty_mlp(
            n_features=input_features,
            lr=uncertainty_lr,
            weight_decay=uncertainty_weight_decay,
            setup_training=True,
            hidden_dim=hidden_dim,
            net_depth=net_depth
        )
        
        print(f"✅ Uncertainty MLP initialized:")
        print(f"   - Input features: {input_features}")
        print(f"   - Hidden dim: {hidden_dim}, Depth: {net_depth}")
        print(f"   - Learning rate: {uncertainty_lr}")
        print(f"   - Weight decay: {uncertainty_weight_decay}")
        
        return uncertainty_mlp
        
    except Exception as e:
        print(f"❌ Failed to initialize uncertainty MLP: {e}")
        return None


def initialize_feature_extractors(config, dataset):
    """
    Initialize feature extractors and depth estimators based on configuration.
    
    Args:
        config: Training configuration dictionary
        dataset: Dataset containing model_path and scene_name
    
    Returns:
        tuple: (feature_extractor, depth_estimator, feature_cfg)
    """
    feature_extractor = None
    depth_estimator = None
    feature_cfg = None
    
    # Check if feature extraction is enabled
    if not config['feature_cfg'].get('enabled', False):
        print("⚠️  Feature extraction disabled in configuration")
        return feature_extractor, depth_estimator, feature_cfg
    
    try:
        # Get feature configuration from config and update dynamic values
        feature_cfg = config['feature_cfg'].copy()
        
        # Models are directly specified in config
            
        # Set dynamic values
        feature_cfg['data']['output'] = dataset.model_path
        if hasattr(dataset, 'scene_name'):
            feature_cfg['scene'] = dataset.scene_name
        
        # Initialize depth estimator for depth-based feature loss
        depth_estimator = get_metric_depth_estimator(feature_cfg)
        print(f"✅ Depth estimator initialized: {feature_cfg['mono_prior']['depth']}")
        
        # Initialize feature extractor only if uncertainty loss is enabled
        if config['uncertainty'].get('enabled', False):
            feature_extractor = get_feature_extractor(feature_cfg)
            print(f"✅ Feature extractor initialized for uncertainty: {feature_cfg['mono_prior']['feature_extractor']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize feature extractors: {e}")
        feature_extractor = None
        depth_estimator = None
        feature_cfg = None
    
    return feature_extractor, depth_estimator, feature_cfg


def prepare_output_and_wandb(args, config):
    """
    Prepare output directories and initialize W&B logging.
    
    Args:
        args: Dataset/model arguments containing model_path
        config: Training configuration dictionary
    
    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    # Set up output folder
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Initialize W&B logging
    wandb_enabled = False
    if WANDB_AVAILABLE:
        try:
            # Create experiment name with date and time
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = getattr(args, 'model_path', 'experiment').split('/')[-1]
            experiment_name = f"{base_name}_{current_time}"
            
            wandb_enabled = init_wandb(
                project_name="StreetUnveiler",
                experiment_name=experiment_name,
                config=config,
                model_path=args.model_path
            )
            
            if wandb_enabled:
                print(f"✅ W&B logging initialized: {experiment_name}")
            else:
                print("⚠️  W&B logging failed to initialize")
                
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            wandb_enabled = False
    else:
        print("⚠️  W&B not available. Install with: pip install wandb")
    
    return wandb_enabled

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
                            if iteration % 10000 == 0 and idx == 0:  # Print only occasionally
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
                                
                                # Log DINO feature cosine similarity if available
                                try:
                                    # Extract DINO features for rendered image
                                    rendered_image_input = image.unsqueeze(0) if image.dim() == 3 else image
                                    rendered_dino_features = predict_img_features(
                                        feature_extractor,
                                        idx,
                                        rendered_image_input,
                                        feature_cfg,
                                        "cuda",
                                        save_feat=False
                                    )
                                    
                                    # Compute cosine similarity using user's formula
                                    if viewpoint.features is not None and rendered_dino_features is not None:
                                        # Reshape for cosine similarity computation
                                        gt_features_flat = viewpoint.features.view(-1, viewpoint.features.shape[-1])
                                        rendered_features_flat = rendered_dino_features.view(-1, rendered_dino_features.shape[-1])
                                        
                                        # Compute cosine similarity
                                        cos_sim = torch.nn.functional.cosine_similarity(
                                            gt_features_flat, rendered_features_flat, dim=-1
                                        )
                                        
                                        # Apply user's transformation: (1 - cos_sim).sub(0.5).div(0.5).clip(0, 1)
                                        dino_cosine_loss = (1 - cos_sim).sub(0.5).div(0.5).clamp(0.0, 1.0)
                                        dino_cosine_loss = dino_cosine_loss.view(viewpoint.features.shape[:2])  # (H, W)
                                        
                                        # Apply jet colormap for visualization
                                        from utils.general_utils import colormap
                                        dino_cosine_colored = colormap(dino_cosine_loss.cpu().numpy(), cmap='jet')
                                        
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/dino_cosine_similarity", dino_cosine_colored, step=iteration, caption=f"DINO Cosine Similarity Loss {viewpoint.image_name}")
                                        
                                except Exception as e:
                                    if iteration % 5000 == 0:  # Only log occasionally to avoid spam
                                        print(f"[WARNING] Failed to compute DINO cosine similarity for validation: {e}")
                                
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
                                    
                                    # Extract DINO features for rendered image for debug
                                    rendered_dino_features_debug = None
                                    if feature_extractor is not None:
                                        try:
                                            rendered_dino_features_debug = predict_img_features(
                                                feature_extractor,
                                                select_frame_id,
                                                rendered_image_3d.unsqueeze(0),
                                                feature_cfg,
                                                "cuda",
                                                save_feat=False
                                            )
                                        except Exception as e:
                                            print(f"[WARNING] Failed to extract DINO features for debug: {e}")
                                    
                                    debug_results = compute_mapping_loss_components(
                                        gt_image,
                                        rendered_image_3d,
                                        ref_depth,
                                        rendered_depth_3d,
                                        uncertainty_vis,  # Use resized uncertainty
                                        opacity_mask,
                                        train_fraction,
                                        ssim_fraction,
                                        uncertainty_cfg,
                                        visibility_mask,
                                        viewpoint.features,  # GT DINO features
                                        rendered_dino_features_debug,  # Rendered DINO features
                                        return_debug_info=True
                                    )
                                    
                                    # Extract debug components with additional depth debug information
                                    uncertainty_loss_map, resized_uncertainty, rgb_l1_loss, depth_l1_loss, depth_mask, small_ssim_loss, small_opacity, small_depth, ssim_loss, rendered_depth_masked, ref_depth_masked, small_depth_loss_before_penalize, small_depth_loss, dino_cosine_similarity, depth_threshold, median_depth = debug_results
                                    
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
                                    
                                    # Log DINO cosine similarity
                                    if dino_cosine_similarity is not None:
                                        dino_cosine_jet = apply_jet_colormap(dino_cosine_similarity)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_dino_cosine_similarity", dino_cosine_jet, step=iteration, caption=f"DINO Cosine Similarity Loss (jet) {viewpoint.image_name}")
                                        
                                        # Log DINO cosine similarity statistics
                                        dino_debug_stats = {
                                            f'debug_dino/{config["name"]}_view_{viewpoint.image_name}_min': dino_cosine_similarity.min().item(),
                                            f'debug_dino/{config["name"]}_view_{viewpoint.image_name}_max': dino_cosine_similarity.max().item(),
                                            f'debug_dino/{config["name"]}_view_{viewpoint.image_name}_mean': dino_cosine_similarity.mean().item(),
                                            f'debug_dino/{config["name"]}_view_{viewpoint.image_name}_std': dino_cosine_similarity.std().item(),
                                        }
                                        log_metrics(dino_debug_stats, step=iteration)
                                    
                                    # ===== DEPTH DEBUG INFORMATION =====
                                    # Log depth threshold and median depth values as scalar metrics
                                    depth_debug_scalars = {
                                        f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_threshold': depth_threshold.item() if hasattr(depth_threshold, 'item') else float(depth_threshold),
                                        f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_median_depth': median_depth.item() if hasattr(median_depth, 'item') else float(median_depth),
                                    }
                                    log_metrics(depth_debug_scalars, step=iteration)
                                    
                                    # Log depth_l1_loss (unmasked) image with jet colormap
                                    if depth_l1_loss is not None:
                                        depth_l1_loss_jet = apply_jet_colormap(depth_l1_loss)
                                        log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_depth_l1_loss", depth_l1_loss_jet, step=iteration, caption=f"Depth L1 Loss (unmasked, jet) {viewpoint.image_name}")
                                    
                                    # Log depth statistics
                                    if ref_depth is not None and rendered_depth_3d is not None:
                                        depth_stats = {
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_ref_depth_min': ref_depth.min().item(),
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_ref_depth_max': ref_depth.max().item(),
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_ref_depth_mean': ref_depth.mean().item(),
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_rendered_depth_min': rendered_depth_3d.min().item(),
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_rendered_depth_max': rendered_depth_3d.max().item(),
                                            f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_rendered_depth_mean': rendered_depth_3d.mean().item(),
                                        }
                                        log_metrics(depth_stats, step=iteration)
                                        
                                        # Log depth difference statistics
                                        if depth_l1_loss is not None:
                                            depth_diff_stats = {
                                                f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_l1_loss_min': depth_l1_loss.min().item(),
                                                f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_l1_loss_max': depth_l1_loss.max().item(),
                                                f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_l1_loss_mean': depth_l1_loss.mean().item(),
                                            }
                                            log_metrics(depth_diff_stats, step=iteration)
                                        
                                        # Log masked depth statistics
                                        if depth_l1_loss is not None and depth_mask is not None:
                                            depth_l1_loss_masked = depth_l1_loss * depth_mask
                                            valid_pixels = depth_mask.sum().item()
                                            if valid_pixels > 0:
                                                masked_depth_stats = {
                                                    f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_l1_loss_masked_mean': depth_l1_loss_masked.sum().item() / valid_pixels,
                                                    f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_mask_valid_pixels': valid_pixels,
                                                    f'debug_depth/{config["name"]}_view_{viewpoint.image_name}_depth_mask_coverage': valid_pixels / depth_mask.numel(),
                                                }
                                                log_metrics(masked_depth_stats, step=iteration)
                                                
                                                # Also log the masked depth L1 loss image
                                                depth_l1_loss_masked_jet = apply_jet_colormap(depth_l1_loss_masked)
                                                log_image(f"{config['name']}_view_{viewpoint.image_name}/debug_depth_l1_loss_masked", depth_l1_loss_masked_jet, step=iteration, caption=f"Depth L1 Loss (masked, jet) {viewpoint.image_name}")
                                        
                                except Exception as e:
                                    print(f"[WARNING] Failed to compute debug info for mapping loss components: {e}")
                        else:
                            # Debug: Print when uncertainty is None
                            if iteration % 1000 == 0 and idx == 0:
                                print(f"[DEBUG] current_uncertainty is None at iteration {iteration}")
                                print(f"[DEBUG] Uncertainty loss enabled: {config['uncertainty'].get('enabled', False)}")
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
