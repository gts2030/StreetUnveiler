#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_semantic
import sys
from scene import Scene, GaussianModel
from scene.env_map import SkyModel
from utils.general_utils import safe_state, requires_grad
from utils.system_utils import mkdir_p
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.semantic_utils import concerned_classes_ind_map, concerned_classes_list, semantic_prob_to_rgb
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.mono_priors.img_feature_extractors import get_feature_extractor, predict_img_features
from utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp, get_uncertainty_and_loss
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, continue_model_path, start_iteration, debug_from):
    start_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    
    # Initialize uncertainty MLP if enabled
    uncertainty_mlp = None
    if hasattr(opt, 'enable_uncertainty_loss') and opt.enable_uncertainty_loss:
        uncertainty_lr = getattr(opt, 'uncertainty_lr', 0.0001)
        uncertainty_mlp = generate_uncertainty_mlp(384, upscale_factor=1, lr=uncertainty_lr, setup_training=True)
        print("Uncertainty MLP initialized")
    
    if continue_model_path:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp, load_iteration=start_iteration)
    else:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp)
    gaussians.training_setup(opt)

    # Initialize depth estimator and feature extractor if needed
    feature_extractor = None
    depth_estimator = None
    if hasattr(opt, 'enable_feature_loss') and opt.enable_feature_loss:
        # Configuration for both depth estimator and feature extractor
        feature_cfg = {
            "device": "cuda",
            "mono_prior": {
                "feature_extractor": getattr(opt, 'feature_extractor_model', 'dinov2_reg_small_fine'),
                "depth": getattr(opt, 'depth_estimator_model', 'metric3d_vit_large')
            },
            "data": {
                "output": dataset.model_path
            },
            "scene": getattr(dataset, 'scene_name', 'default_scene'),
            "cam": {
                "fx": 1000.0  # Default focal length, should be adjusted based on dataset
            }
        }
        
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
    opt.densification_interval = int(len(scene.getTrainCameras()) * 1.15)
    print(opt.densification_interval)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.prune_semantic_splatting(1 << concerned_classes_ind_map['sky'])
    
    # Store uncertainty for visualization
    current_uncertainty = None
    
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
            semantic_loss = F.cross_entropy(render_semantics.unsqueeze(0), gt_semantic.unsqueeze(0), weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.2, 1.0]).cuda())

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
            
            # Predict depth from ground truth image
            gt_depth = predict_metric_depth(
                depth_estimator,
                select_frame_id,
                gt_image_input,
                feature_cfg,
                "cuda",
                save_depth=True  # Don't save during training
            )
            
            # Get rendered depth
            rendered_depth = render_pkg.get("surf_depth", None)
            if rendered_depth is not None:
                # Ensure both depths have the same shape
                if rendered_depth.dim() > 2:
                    rendered_depth = rendered_depth.squeeze()
                if gt_depth.dim() > 2:
                    gt_depth = gt_depth.squeeze()
                
                # Resize if necessary
                if rendered_depth.shape != gt_depth.shape:
                    gt_depth = F.interpolate(
                        gt_depth.unsqueeze(0).unsqueeze(0), 
                        size=rendered_depth.shape, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                
                # Compute depth L1 loss
                feature_loss = F.l1_loss(rendered_depth, gt_depth) * feature_loss_weight
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
                lambda1 = getattr(opt, 'uncertainty_lambda1', 1.0)
                lambda2 = getattr(opt, 'uncertainty_lambda2', 0.01)
                lambda3 = getattr(opt, 'uncertainty_lambda3', 0.01)
                
                # Check if depth is available for depth uncertainty term
                depth_rendered = render_pkg.get("surf_depth", None)
                depth_gt = getattr(viewpoint_cam, 'depth', None)
                if depth_gt is not None:
                    depth_gt = torch.from_numpy(depth_gt).cuda().float()
                
                # Use detached losses for uncertainty prediction (no gradient to Gaussians)
                uncertainty_Ll1 = Ll1.detach()
                uncertainty_ssim_map = ssim_map.detach()
                uncertainty_depth_rendered = depth_rendered.detach() if depth_rendered is not None else None
                
                uncertainty, uncertainty_loss = get_uncertainty_and_loss(
                    uncertainty_features,
                    scene.uncertainty_mlp,
                    uncertainty_Ll1,  # Detached L1 loss
                    uncertainty_ssim_map,  # Detached SSIM map
                    uncertainty_depth_rendered,  # Detached depth
                    depth_gt,
                    lambda1,
                    lambda2,
                    lambda3
                )
                
                # Store uncertainty for visualization and render loss computation
                current_uncertainty = uncertainty.detach()
                
                # Train uncertainty MLP independently (as per paper)
                loss_dict['uncertainty'] = uncertainty_loss
                uncertainty_loss.backward()  # Uncertainty MLP만 업데이트

        # Compute render loss with uncertainty weighting if available
        # L_render = (λ5*L_color + λ6*L_depth) / β^2 + λ7*L_iso
        if current_uncertainty is not None:
            # Get render loss parameters
            lambda5 = getattr(opt, 'render_lambda5', 1.0)  # Color loss weight
            lambda6 = getattr(opt, 'render_lambda6', 1.0)  # Depth loss weight  
            lambda7 = getattr(opt, 'render_lambda7', 0.01)  # Isotropic regularization weight
            
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
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), sky_model, current_uncertainty)
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

    # Generate uncertainty maps for all input images at the end of training
    if scene.uncertainty_mlp is not None and feature_extractor is not None:
        print("\nGenerating uncertainty maps for all input images...")
        uncertainty_output_dir = os.path.join(scene.model_path, "uncertainty_maps")
        os.makedirs(uncertainty_output_dir, exist_ok=True)
        
        # Process all training cameras
        with torch.no_grad():
            scene.uncertainty_mlp.eval()
            for idx, viewpoint_cam in enumerate(tqdm(scene.getTrainCameras(), desc="Generating uncertainty maps")):
                try:
                    # Use ground truth image for uncertainty prediction
                    gt_image = viewpoint_cam.original_image.cuda()
                    if gt_image.dim() == 3:
                        gt_image_input = gt_image.unsqueeze(0)
                    else:
                        gt_image_input = gt_image
                    
                    # Extract features (same as training)
                    uncertainty_features = predict_img_features(
                        feature_extractor,
                        idx,
                        gt_image_input,
                        feature_cfg,
                        "cuda",
                        save_feat=False
                    )
                    
                    if uncertainty_features is not None:
                        # Predict uncertainty
                        uncertainty = scene.uncertainty_mlp(uncertainty_features)
                        
                        # Resize to match image dimensions if needed
                        target_size = gt_image.shape[-2:]
                        if uncertainty.shape != target_size:
                            uncertainty = F.interpolate(
                                uncertainty.unsqueeze(0).unsqueeze(0) if uncertainty.dim() == 2 else uncertainty.unsqueeze(0),
                                size=target_size,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        
                        # Save both raw data and visualization
                        frame_name = f"{idx:05d}"
                        if hasattr(viewpoint_cam, 'image_name'):
                            frame_name = viewpoint_cam.image_name
                        
                        # Save raw uncertainty (.npy)
                        uncertainty_np = uncertainty.detach().cpu().numpy()
                        np.save(os.path.join(uncertainty_output_dir, f"{frame_name}_uncertainty.npy"), uncertainty_np)
                        
                        # Save visualization (.png)
                        uncertainty_norm = (uncertainty_np - uncertainty_np.min()) / (uncertainty_np.max() - uncertainty_np.min() + 1e-8)
                        uncertainty_vis = (uncertainty_norm * 255).astype(np.uint8)
                        
                        # Apply jet colormap
                        import cv2
                        uncertainty_colored = cv2.applyColorMap(uncertainty_vis, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(uncertainty_output_dir, f"{frame_name}_uncertainty.png"), uncertainty_colored)
                        
                except Exception as e:
                    print(f"Failed to generate uncertainty map for frame {idx}: {e}")
        
        print(f"Uncertainty maps saved to: {uncertainty_output_dir}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(os.path.join(scene.model_path, "checkpoint", "computation_statistics.txt"), 'w', encoding='utf-8') as file:
        file.write("2DGS training {} seconds.".format(elapsed_time))

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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, sky_model, current_uncertainty=None):
    if tb_writer:
        for key, value in loss_dict.items():
            tb_writer.add_scalar('train_loss_patches/{}_loss'.format(key), value.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/sky".format(viewpoint.image_name), env_image[None], global_step=iteration)
                        from utils.general_utils import colormap
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/disparity".format(viewpoint.image_name), disparity[None], global_step=iteration)
                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                             rend_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                             surf_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                             rend_alpha[None], global_step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                             rend_dist[None], global_step=iteration)

                        semantic_pkg = render_semantic(viewpoint, scene.gaussians, *renderArgs)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_semantic".format(viewpoint.image_name),
                                             semantic_pkg['semantic_rgb'][None], global_step=iteration)
                        
                        # Add uncertainty heat map visualization
                        if current_uncertainty is not None:
                            # Resize uncertainty to match image dimensions if needed
                            uncertainty_vis = current_uncertainty
                            if uncertainty_vis.shape != (viewpoint.image_height, viewpoint.image_width):
                                uncertainty_vis = F.interpolate(
                                    uncertainty_vis.unsqueeze(0).unsqueeze(0),
                                    size=(viewpoint.image_height, viewpoint.image_width),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze()
                            
                            # Convert to heat map using colormap
                            uncertainty_heatmap = colormap(uncertainty_vis.cpu().numpy(), cmap='jet')
                            tb_writer.add_images(config['name'] + "_view_{}/uncertainty_heatmap".format(viewpoint.image_name),
                                                 uncertainty_heatmap[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/semantic_gt".format(viewpoint.image_name), semantic_prob_to_rgb(viewpoint.get_semantic_prob_image())[None] / 255., global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.continue_model_path, args.start_iteration, args.debug_from)

    # All done
    print("\nTraining complete.")
