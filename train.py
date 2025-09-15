
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
from alive_progress import alive_bar
from utils.image_utils import psnr
from utils.semantic_utils import concerned_classes_ind_map, concerned_classes_list, semantic_prob_to_rgb
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.wandb_utils import prepare_output_and_wandb, init_wandb, log_scalar, log_image, log_histogram, log_metrics, finish_wandb, is_wandb_available
from utils.mono_priors.metric_depth_estimators import compute_metric_depth, get_metric_depth_estimator
from utils.mono_priors.img_feature_extractors import get_feature_extractor, predict_img_features
from utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, compute_metric_depth
from utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from utils.dyn_uncertainty.mapping_utils import compute_mapping_loss_components, compute_dino_regularization_loss
from PIL import Image

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
                gt_in = (gt_image.unsqueeze(0) if gt_image.dim()==3 else gt_image)
                viewpoint_cam.features = predict_img_features(feature_extractor, cam_idx, gt_in, dataset, save_feat=False)
            
            # Predict uncertainty
            beta_pred = scene.uncertainty_mlp(viewpoint_cam.features)
            
            # Apply the same clamping as in training
            beta_pred = torch.clamp(beta_pred, min=0.5, max=5.0)
            
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
            
            train_frac = 1.0  # Use full training fraction for final iteration
            ssim_frac = 1.0
            
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
            
            # Convert resized_uncertainty to numpy for saving
            uncertainty_np = beta_resized.detach().cpu().numpy()
            
            # Save uncertainty with colormap only
            from utils.general_utils import colormap
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
        print("⚠️ depth estimator is None, skip GT depth precompute"); return
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, continue_model_path, start_iteration, debug_from):
    start_time = time.time()
    first_iter = 0

    # Initialize W&B logging
    wandb_enabled = prepare_output_and_wandb(dataset)
    if not wandb_enabled:
        print("W&B logging failed to initialize. Continuing without logging.")
    
    # Initialize depth estimator once for efficiency (only if needed)
    depth_estimator = get_metric_depth_estimator(dataset) if opt.use_depth_loss_in_uncertainty else None
    
    # Initialize DINO feature extractor once for efficiency
    feature_extractor = get_feature_extractor(dataset)
    
    # Initialize uncertainty MLP
    uncertainty_mlp = _init_uncertainty_mlp(opt)

    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    
    if continue_model_path:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp, load_iteration=start_iteration)
    else:
        scene = Scene(dataset, gaussians, sky_model, uncertainty_mlp)
    gaussians.training_setup(opt)

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
    print("Densification interval: ", opt.densification_interval)
    ema_loss_for_log = 0.0
    total_iterations = opt.iterations - first_iter
    first_iter += 1
    gaussians.prune_semantic_splatting(1 << concerned_classes_ind_map['sky'])

    # precompute GT depths if depth loss is enabled
    if opt.use_depth_loss_in_uncertainty:
        _precompute_gt_depths(scene, depth_estimator, dataset)
    else:
        scene.computed_gt_depths = {}
    
    with alive_bar(total_iterations, title="🚀 Training Dynamic StreetUnveiler", bar="smooth", spinner="waves", file=sys.stderr) as bar:
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

            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            render_image, viewspace_point_tensor, visibility_filter, radii, opacity, surf_depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rend_alpha"], render_pkg["surf_depth"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_metric_depth = scene.computed_gt_depths.get(select_frame_id) if opt.use_depth_loss_in_uncertainty else None
            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            composite_image = render_image + sky_image * (1 - opacity)
            Ll1 = l1_loss(composite_image, gt_image)
            Lssim = ssim(composite_image, gt_image)
            
            # Conditionally compute metric depth based on argument
            if opt.use_depth_loss_in_uncertainty:
                rendered_metric_depth = compute_metric_depth(
                    depth_estimator=depth_estimator,
                    frame_id=select_frame_id,
                    image_input=composite_image,
                    feature_cfg=None,
                    rendered_depth=None,
                    viewpoint_cam=viewpoint_cam,
                    dataset=dataset
                )
            else:
                rendered_metric_depth = None

            # Extract DINO features from gt image and cache it
            if not hasattr(viewpoint_cam, 'features') or viewpoint_cam.features is None:
                gt_in = (gt_image.unsqueeze(0) if gt_image.dim()==3 else gt_image)
                viewpoint_cam.features = predict_img_features(feature_extractor, select_frame_id, gt_in, dataset, save_feat=False)
            
            # Extract DINO features from rendered image
            composite_in = composite_image.unsqueeze(0) if composite_image.dim() == 3 else composite_image
            rendered_features = predict_img_features(feature_extractor, select_frame_id, composite_in, dataset, save_feat=False)

            train_frac = min(1.0, iteration / float(opt.iterations))
            ssim_frac  = train_frac
            beta_pred = scene.uncertainty_mlp(viewpoint_cam.features) if scene.uncertainty_mlp is not None else torch.ones_like(surf_depth.squeeze(0))
            
            # Prevent extremely small/large beta values that cause gradient explosion/instability
            if scene.uncertainty_mlp is not None:
                beta_pred = torch.clamp(beta_pred, min=0.5, max=5.0)
            
            # Get opacity mask
            opacity_mask = render_pkg.get("opacity", torch.ones(gt_image.shape[-2:], device=gt_image.device))
            if opacity_mask.dim() == 2:
                opacity_mask = opacity_mask.unsqueeze(0)
            
            # Compute loss components
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

            uncertainty_loss = uncer_loss_map.mean()
            if scene.uncertainty_mlp is not None and hasattr(viewpoint_cam, 'features') and viewpoint_cam.features is not None and opt.lambda_dino_reg > 0:
                reg_v = compute_dino_regularization_loss(beta_pred, viewpoint_cam.features)
                uncertainty_loss = uncertainty_loss + opt.lambda_dino_reg * reg_v

            # Use predicted uncertainty directly without schedule
            beta_resized = beta_resized.clamp_min(1e-3)  # safety clamp
            beta2 = (beta_resized ** 2).clamp_min(1e-8)

            # render loss with predicted beta
            rgb_l1_perpix = rgb_l1_map.mean(0)          # [H,W]
            render_Lssim = ssim(composite_image, gt_image)
            L_color = (1.0 - opt.lambda_dssim) * (rgb_l1_perpix / beta2).mean() + opt.lambda_dssim * (1.0 - render_Lssim)
            
            # depth_l1_perpix = depth_l1_map.squeeze(0)     # [H,W]
            # render_loss_uncer = (
            #     opt.lambda_rgb_l1   *  L_color +
            #     opt.lambda_depth_l1 * (depth_l1_perpix / beta2).mean()
            # )
            loss = L_color
            loss_dict['Lrender_uncer'] = L_color
            loss_dict['beta_resized'] = beta_resized

            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
            # loss_dict['l1'] = Ll1
            # loss_dict['ssim'] = Lssim

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

            loss.backward()
            
            # Clip gradients to prevent explosion
            # For GaussianModel, clip gradients for each parameter group
            for group in gaussians.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            
            # Train uncertainty MLP from the beginning
            if scene.uncertainty_mlp is not None:
                uncertainty_loss.backward()
                # Clip uncertainty MLP gradients as well
                torch.nn.utils.clip_grad_norm_(scene.uncertainty_mlp.parameters(), max_norm=1.0)
                scene.step_uncertainty_optimizer()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    bar.text = f"Loss: {ema_loss_for_log:.7f}"
                    for _ in range(10):
                        bar()

                # Log and save
                training_report(iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), sky_model, dataset, depth_estimator, feature_extractor, opt)
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

                if (iteration in checkpoint_iterations):
                    checkpoint_path = os.path.join(scene.model_path, "checkpoint", "iteration_{}".format(iteration))
                    mkdir_p(checkpoint_path)
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), os.path.join(checkpoint_path, "splatting.pt"))
                    sky_model.save(os.path.join(checkpoint_path, "sky_params.pt"))

    # Save uncertainty images for all training cameras at the last iteration
    if iteration == opt.iterations:
        _save_all_uncertainty_images(scene, gaussians, sky_model, pipe, background, dataset, depth_estimator, feature_extractor, opt, iteration)

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

    # W&B logging is handled by wandb_utils
    return None

def training_report(iteration, loss_dict, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, sky_model, dataset=None, depth_estimator=None, feature_extractor=None, opt=None):
    # Log metrics to W&B
    if is_wandb_available():
        metrics = {}
        for key, value in loss_dict.items():
            if key == 'beta_resized':
                # Log beta_resized as image with colormap
                from utils.general_utils import colormap_no_bar
                beta_resized_colormap = colormap_no_bar(value.detach().cpu().numpy())
                log_image(f'train_images/beta_resized', beta_resized_colormap, step=iteration)
            else:
                metrics[f'train_loss_patches/{key}_loss'] = value.item()
        metrics['train_loss_patches/total_loss'] = loss.item()
        metrics['iter_time'] = elapsed
        log_metrics(metrics, step=iteration)

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
                first_debug_scalars = None  # Store scalars from first camera
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    select_frame_id = pick_frame_list[idx]
                    env_image = sky_model.render_with_camera(viewpoint.image_height, viewpoint.image_width, viewpoint.K, viewpoint.c2w)
                    image = torch.clamp(render_pkg["render"] + (1.0 - render_pkg['rend_alpha']) * env_image, 0.0, 1.0)
                    disparity = torch.clamp((1.0 / render_pkg["surf_depth"]).nan_to_num(), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Compute metric depth for validation (only if needed)
                    if opt.use_depth_loss_in_uncertainty and depth_estimator is not None:
                        rendered_metric_depth = compute_metric_depth(
                            depth_estimator=depth_estimator,
                            frame_id=select_frame_id,
                            image_input=image,
                            feature_cfg={},
                            rendered_depth=render_pkg.get("surf_depth"),
                            viewpoint_cam=viewpoint,
                            dataset=dataset
                        )

                        gt_metric_depth = compute_metric_depth(
                            depth_estimator=depth_estimator,
                            frame_id=select_frame_id,
                            image_input=gt_image,
                            feature_cfg={},
                            rendered_depth=render_pkg.get("surf_depth"),
                            viewpoint_cam=viewpoint,
                            dataset=dataset
                        )
                    else:
                        rendered_metric_depth = None
                        gt_metric_depth = None
                    
                    # Compute mapping loss components in debug mode for logging
                    if scene.uncertainty_mlp is not None and hasattr(viewpoint, 'features') and viewpoint.features is not None:
                        image_in = image.unsqueeze(0) if image.dim() == 3 else image
                        rendered_features = predict_img_features(feature_extractor, select_frame_id, image_in, dataset, save_feat=False)
                        train_frac = min(1.0, iteration / float(1000))  # Use a reasonable train fraction for validation
                        ssim_frac = train_frac
                        beta_pred = scene.uncertainty_mlp(viewpoint.features)
                        
                        # Get opacity mask for debug info
                        debug_opacity_mask = render_pkg.get("rend_alpha", torch.ones(gt_image.shape[-2:], device=gt_image.device))
                        if debug_opacity_mask.dim() == 2:
                            debug_opacity_mask = debug_opacity_mask.unsqueeze(0)
                        
                        # Use dummy depth if depth loss is disabled
                        debug_gt_depth = gt_metric_depth if gt_metric_depth is not None else torch.ones_like(render_pkg["surf_depth"])
                        debug_rendered_depth = rendered_metric_depth if rendered_metric_depth is not None else torch.ones_like(render_pkg["surf_depth"])
                        
                        # Get all debug info from mapping loss components
                        debug_results = compute_mapping_loss_components(
                            gt_img=gt_image, rendered_img=image,
                            ref_depth=debug_gt_depth, rendered_depth=debug_rendered_depth,
                            uncertainty=beta_pred, opacity=render_pkg["rend_alpha"],
                            train_fraction=train_frac, ssim_fraction=ssim_frac,
                            opt=opt,
                            mask=debug_opacity_mask,
                            gt_dino_features=viewpoint.features,
                            rendered_dino_features=rendered_features,
                            return_debug_info=True
                        )
                        
                        # Extract debug variables for logging
                        (uncertainty_loss_map, resized_uncertainty, rgb_l1_loss_map, depth_l1_loss_map, 
                         depth_mask, small_ssim_loss, small_opacity, small_depth, ssim_loss,
                         rendered_depth_masked, ref_depth_masked, small_depth_loss_before_penalize, 
                         small_depth_loss, dino_cosine_similarity, filtered_ssim_loss_min, depth_threshold, median_depth) = debug_results
                        
                        # Store scalar values from first camera for later logging (only if depth loss is enabled)
                        if idx == 0 and opt.use_depth_loss_in_uncertainty:
                            first_debug_scalars = {
                                'depth_threshold': depth_threshold,
                                'median_depth': median_depth
                            }
                    
                    if is_wandb_available() and (idx < 8):
                        from utils.general_utils import colormap_no_bar, colormap
                        log_image(config['name'] + "_view_{}/sky".format(viewpoint.image_name), env_image, step=iteration)
                        log_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, step=iteration)
                        log_image(config['name'] + "_view_{}/disparity".format(viewpoint.image_name), disparity, step=iteration)
                        
                        # Log surf_depth with colormap
                        surf_depth_colormap = colormap(render_pkg["surf_depth"].cpu().numpy()[0])
                        log_image(config['name'] + "_view_{}/surf_depth".format(viewpoint.image_name), surf_depth_colormap, step=iteration)
                        
                        # Log opacity_mask
                        opacity_mask = render_pkg.get("opacity", torch.ones(gt_image.shape[-2:], device=gt_image.device))
                        if opacity_mask.dim() == 2:
                            opacity_mask = opacity_mask.unsqueeze(0)
                        log_image(config['name'] + "_view_{}/opacity_mask".format(viewpoint.image_name), opacity_mask, step=iteration)
                        
                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        log_image(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal, step=iteration)
                        log_image(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal, step=iteration)
                        log_image(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha, step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap_no_bar(rend_dist.cpu().numpy()[0])
                        log_image(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist, step=iteration)

                        semantic_pkg = render_semantic(viewpoint, scene.gaussians, *renderArgs)
                        log_image(config['name'] + "_view_{}/rend_semantic".format(viewpoint.image_name), semantic_pkg['semantic_rgb'], step=iteration)

                        # Log metric depth images with colormap (only if depth loss is enabled)
                        if opt.use_depth_loss_in_uncertainty:
                            if rendered_metric_depth is not None:
                                # Apply colormap like rend_dist for better visualization
                                rendered_depth_colormap = colormap(rendered_metric_depth.cpu().numpy())
                                log_image(config['name'] + "_view_{}/rendered_metric_depth".format(viewpoint.image_name), rendered_depth_colormap, step=iteration)
                            
                            if gt_metric_depth is not None:
                                # Apply colormap like rend_dist for better visualization
                                gt_depth_colormap = colormap(gt_metric_depth.cpu().numpy())
                                log_image(config['name'] + "_view_{}/gt_metric_depth".format(viewpoint.image_name), gt_depth_colormap, step=iteration)

                        # Log debug variables from mapping loss components (images only, excluding scalar values)
                        if 'debug_results' in locals():
                            # Log 2D tensor images with appropriate normalization and colormap
                            debug_image_vars = {
                                'uncertainty_loss_map': uncertainty_loss_map,
                                'resized_uncertainty': resized_uncertainty,
                                'uncertainty_feature': beta_pred,  # Add beta_pred as uncertainty_feature
                                'rgb_l1_loss_map': rgb_l1_loss_map.mean(0) if len(rgb_l1_loss_map.shape) > 2 else rgb_l1_loss_map,  # Average RGB channels if needed
                                'small_ssim_loss': small_ssim_loss,
                                'small_opacity': small_opacity,
                                'ssim_loss': ssim_loss,
                                'filtered_ssim_loss_min': filtered_ssim_loss_min
                            }
                            
                            # Add depth-related debug variables only if depth loss is enabled
                            if opt.use_depth_loss_in_uncertainty:
                                debug_image_vars.update({
                                    'depth_l1_loss_map': depth_l1_loss_map.squeeze() if len(depth_l1_loss_map.shape) > 2 else depth_l1_loss_map,
                                    'depth_mask': depth_mask.squeeze().float() if len(depth_mask.shape) > 2 else depth_mask.float(),
                                    'small_depth': small_depth,
                                    'rendered_depth_masked': rendered_depth_masked.squeeze() if len(rendered_depth_masked.shape) > 2 else rendered_depth_masked,
                                    'ref_depth_masked': ref_depth_masked.squeeze() if len(ref_depth_masked.shape) > 2 else ref_depth_masked,
                                    'small_depth_loss_before_penalize': small_depth_loss_before_penalize,
                                    'small_depth_loss': small_depth_loss,
                                })
                            
                            # Add dino_cosine_similarity if it's not None
                            if dino_cosine_similarity is not None:
                                debug_image_vars['dino_cosine_similarity'] = dino_cosine_similarity
                            
                            # Log each debug image with proper colormap
                            # Variables that benefit from colorbar (for value range understanding)
                            colorbar_useful_vars = ['uncertainty_loss_map', 'resized_uncertainty', 'uncertainty_feature']
                            if opt.use_depth_loss_in_uncertainty:
                                colorbar_useful_vars.extend(['rendered_depth_masked', 'ref_depth_masked', 'small_depth'])
                            
                            for var_name, var_tensor in debug_image_vars.items():
                                if var_tensor is not None and len(var_tensor.shape) >= 2:
                                    # Handle different tensor shapes and apply colormap for better visualization
                                    if var_name in ['depth_mask']:
                                        # Binary mask - no colormap needed
                                        log_image(config['name'] + "_view_{}/debug_{}".format(viewpoint.image_name, var_name), var_tensor, step=iteration)
                                    elif var_name in colorbar_useful_vars:
                                        # Apply colormap with colorbar for depth/uncertainty variables
                                        var_colormap = colormap(var_tensor.detach().cpu().numpy())
                                        log_image(config['name'] + "_view_{}/debug_{}".format(viewpoint.image_name, var_name), var_colormap, step=iteration)
                                    else:
                                        # Apply colormap without colorbar for other tensors
                                        var_colormap = colormap_no_bar(var_tensor.detach().cpu().numpy())
                                        log_image(config['name'] + "_view_{}/debug_{}".format(viewpoint.image_name, var_name), var_colormap, step=iteration)

                        if iteration == testing_iterations[0]:
                            log_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, step=iteration)
                            log_image(config['name'] + "_view_{}/semantic_gt".format(viewpoint.image_name), semantic_prob_to_rgb(viewpoint.get_semantic_prob_image()) / 255., step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if is_wandb_available():
                    log_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, step=iteration)
                    log_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, step=iteration)
                    
                    # Log debug scalar values (depth_threshold and median_depth) once per config (only if depth loss is enabled)
                    if opt.use_depth_loss_in_uncertainty and 'first_debug_scalars' in locals() and first_debug_scalars is not None and config['name'] == 'test':  # Log only for test config to avoid duplication
                        log_scalar('debug_scalars/depth_threshold', 
                                 first_debug_scalars['depth_threshold'].item() if hasattr(first_debug_scalars['depth_threshold'], 'item') else first_debug_scalars['depth_threshold'], 
                                 step=iteration)
                        log_scalar('debug_scalars/median_depth', 
                                 first_debug_scalars['median_depth'].item() if hasattr(first_debug_scalars['median_depth'], 'item') else first_debug_scalars['median_depth'], 
                                 step=iteration)

        if is_wandb_available():
            log_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, step=iteration)
            log_scalar('total_points', scene.gaussians.get_xyz.shape[0], step=iteration)
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
