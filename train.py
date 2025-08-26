
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, continue_model_path, start_iteration, debug_from):
    start_time = time.time()
    first_iter = 0

    # Initialize W&B logging
    wandb_enabled = prepare_output_and_wandb(dataset)
    if not wandb_enabled:
        print("W&B logging failed to initialize. Continuing without logging.")

    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    
    # Initialize depth estimator once for efficiency
    depth_estimator = get_metric_depth_estimator(dataset=dataset)
    
    if continue_model_path:
        scene = Scene(dataset, gaussians, sky_model, load_iteration=start_iteration)
    else:
        scene = Scene(dataset, gaussians, sky_model)
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
    print(opt.densification_interval)
    ema_loss_for_log = 0.0
    total_iterations = opt.iterations - first_iter
    first_iter += 1
    gaussians.prune_semantic_splatting(1 << concerned_classes_ind_map['sky'])
    
    import sys
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
            render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            composite_image = render_image + sky_image * (1 - render_pkg["rend_alpha"])
            Ll1 = l1_loss(composite_image, gt_image)
            Lssim = ssim(composite_image, gt_image)
            
            # Example: Compute metric depth for current viewpoint (if needed for depth supervision)
            # metric_depth = compute_metric_depth(
            #     depth_estimator=depth_estimator,
            #     frame_id=iteration,
            #     image_input=gt_image,
            #     feature_cfg={},
            #     rendered_depth=render_pkg.get("surf_depth"),
            #     viewpoint_cam=viewpoint_cam,
            #     dataset=dataset
            # )

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)

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

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    bar.text = f"Loss: {ema_loss_for_log:.7f}"
                    for _ in range(10):
                        bar()

                # Log and save
                training_report(iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), sky_model, dataset, depth_estimator)
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

def training_report(iteration, loss_dict, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, sky_model, dataset=None, depth_estimator=None):
    # Log metrics to W&B
    if is_wandb_available():
        metrics = {}
        for key, value in loss_dict.items():
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
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    select_frame_id = pick_frame_list[idx]
                    env_image = sky_model.render_with_camera(viewpoint.image_height, viewpoint.image_width, viewpoint.K, viewpoint.c2w)
                    image = torch.clamp(render_pkg["render"] + (1.0 - render_pkg['rend_alpha']) * env_image, 0.0, 1.0)
                    disparity = torch.clamp((1.0 / render_pkg["surf_depth"]).nan_to_num(), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Compute metric depth for validation
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
                    
                    if is_wandb_available() and (idx < 8):
                        from utils.general_utils import colormap
                        log_image(config['name'] + "_view_{}/sky".format(viewpoint.image_name), env_image, step=iteration)
                        log_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, step=iteration)
                        log_image(config['name'] + "_view_{}/disparity".format(viewpoint.image_name), disparity, step=iteration)
                        
                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                        log_image(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal, step=iteration)
                        log_image(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal, step=iteration)
                        log_image(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha, step=iteration)

                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        log_image(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist, step=iteration)

                        semantic_pkg = render_semantic(viewpoint, scene.gaussians, *renderArgs)
                        log_image(config['name'] + "_view_{}/rend_semantic".format(viewpoint.image_name), semantic_pkg['semantic_rgb'], step=iteration)

                        # Log metric depth images with colormap
                        if rendered_metric_depth is not None:
                            # Apply colormap like rend_dist for better visualization
                            rendered_depth_colormap = colormap(rendered_metric_depth.cpu().numpy())
                            log_image(config['name'] + "_view_{}/rendered_metric_depth".format(viewpoint.image_name), rendered_depth_colormap, step=iteration)
                        
                        if gt_metric_depth is not None:
                            # Apply colormap like rend_dist for better visualization
                            gt_depth_colormap = colormap(gt_metric_depth.cpu().numpy())
                            log_image(config['name'] + "_view_{}/gt_metric_depth".format(viewpoint.image_name), gt_depth_colormap, step=iteration)

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
