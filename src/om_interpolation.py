"""
Interpolation via Onsager-Machlup action minimization of a diffusion model. 
We do forward diffusion for two images and then start with a linear/spherical interpolation path as an initial guess.
Then, we optimize the path by minimizing the Onsager-Machlup action.
Finally, we sample from the model via reverse diffusion along the optimized latent path to get the corresponding image.
We can interpolate in image space or in pure Gaussian noise space by setting the latent_time parameter.
We can also default to vanilla linear or spherical interpolation by setting the steps parameter to 0.
"""

import math
from datetime import datetime
import numpy as np
import argparse
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb


wandb.require("core")
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from data.dataloaders import create_mnist_dataloaders, create_celeba_dataloaders

from model import MNISTDiffusion, CelebADiffusion
from actions import SimpleAction, TruncatedAction, HessianAction
from utils import get_initial_guess_fn, validate_git_status

from torcheval.metrics import FrechetInceptionDistance
from metrics import perceptual_path_length_and_variance
from lpips import LPIPS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--cpu", action="store_true", help="cpu training")
    parser.add_argument(
        "--disable_logging", action="store_true", help="disable wandb logging"
    )
    parser.add_argument(
        "--exp_name", type=str, default="om_interpolation", help="wandb experiment name"
    )
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument(
        "--ckpt_path", type=str, help="define checkpoint path", default="best_model.pt"
    )
    parser.add_argument(
        "--latent_time",
        type=int,
        help="number of timesteps for forward/reverse diffusion (i.e at what point in \
        the diffusion process to do interpolation). Must be <= 1000. 0 corresponds to \
        interpolation in image space, and 1000 corresponds to interpolation in pure Gaussian noise space.",
        # default=300,
        # latent time 0 is good for the start. Lets see if it helps to use something more.
        default=0,
    )

    parser.add_argument(
        "--initial_guess_method",
        type=str,
        help="method to generate initial interpolation path (options: 'spherical' or 'linear')",
        default="linear",
    )

    parser.add_argument(
        "--path_length", type=int, help="length of interpolation path", default=16
    )
    parser.add_argument(
        "--steps", type=int, help="number of OM optimization steps", default=1000
    )
    parser.add_argument(
        "--const_time", type=float, help="timestep for OM action", default=4.0
    )

    parser.add_argument(
        "--noise_perturb_scale",
        type=float,
        help="scale of noise perturbing OM path after every optimization step",
        default=0.00,
    )

    parser.add_argument(
        "--lr", type=float, help="learning rate for OM optimization", default=1e-2
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        help="number of data pairs to do interpolation with",
        default=1000,
    )
    parser.add_argument(
        "--save_every", type=int, help="save images every n steps", default=50
    )

    parser.add_argument(
        "--action",
        type=str,
        help="Which action to use. Options: hessian, truncated, simple",
        default="truncated",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch_size for doing the optimization",
        default=96,  # this saturates GPU memory on Sanjeev's Germain server for path length of 16
    )

    parser.add_argument(
        "--v_scale",
        type=float,
        help="how much to scale vector field for action",
        default=1.0,
    )

    parser.add_argument(
        "--kernel_var",
        type=float,
        help="variance for gaussian kernel for path norm. setting to 0 uses no gaussian kernel convolution",
        default=0.0,
    )

    parser.add_argument(
        "--truncate_v_gradient",
        action="store_true",
        help="approx gradient that doesn't go through diffusion model. essentially integrates path along vector field from diffusion model",
    )

    parser.add_argument(
        "--steps_v",
        type=int,
        help="how many diffusion steps to compute vector field. NOTE: only enabled if truncate_v_gradient is enabled",
        default=1,  # this saturates GPU memory on Sanjeev's Germain server for path length of 16
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset (and corresponding trained model) to use. Options: mnist, celeba",
        default="mnist",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which directory to load datasets from",
        default="data",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu" if args.cpu else "cuda"

    # If using Martins poor man's GPU we need to lower the batch size.

    poor_mans_gpu = torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1660 Ti"

    if poor_mans_gpu:
        print("Lowering batch size")
        args.batch_size = 2
        # args.max_pairs = 3

    if not args.disable_logging:
        validate_git_status()
        wandb.login()
        wandb.init(project="om-diffusion", config=args, name=args.exp_name)

    # Get arguments
    ckpt_path = args.ckpt_path
    path_length = args.path_length
    latent_time = args.latent_time
    if latent_time >= 1000:
        raise ValueError(
            "latent_time must be less than total diffusion model time of 1000"
        )
    initial_guess_method = args.initial_guess_method
    if initial_guess_method not in ["spherical", "linear"]:
        raise ValueError("initial_guess_method must be 'spherical' or 'linear'")
    initial_guess_fn = get_initial_guess_fn(initial_guess_method)
    steps = args.steps
    max_pairs = args.max_pairs
    save_every = args.save_every
    const_time = args.const_time
    noise_perturb_scale = args.noise_perturb_scale
    lr = args.lr
    batch_size = args.batch_size
    max_batches = math.ceil(max_pairs / batch_size)

    v_scale = args.v_scale
    v_steps = args.steps_v
    g_sigma = args.kernel_var

    t = torch.tensor(latent_time).to(device)

    os.makedirs("../mnist_outputs/", exist_ok=True)

    # Instantiate dataloader
    if args.dataset == "mnist":
        train_dataloader, test_dataloader = create_mnist_dataloaders(
            batch_size=batch_size
        )
    elif args.dataset == "celeba":
        train_dataloader, test_dataloader = create_celeba_dataloaders(
            batch_size=batch_size, root_dir=args.data_dir
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized")

    train_iterator = iter(train_dataloader)
    test_iterator = iter(test_dataloader)

    init_im, _ = next(test_iterator)
    final_im, _ = next(test_iterator)

    _, C, H, W = init_im.shape

    # Un-normalize images from [-1, 1]  to [0, 1]
    inv_normalizer = transforms.Normalize([-1.0], [2.0])

    if poor_mans_gpu:
        in_channels = 1
        time_embedding_dim = 256
        timesteps = 1000
        base_dim = 64
        dim_mults = [2, 4]

        model = MNISTDiffusion(
            28,
            in_channels,
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
        )

        ckpt = torch.load("data/models/best_model.pt")
        model.load_state_dict(ckpt["model"])

    else:

        # Instantiate model
        in_channels = 1
        time_embedding_dim = 256
        timesteps = 1000
        base_dim = 64
        dim_mults = [2, 4]

        if args.dataset == "mnist":
            model = MNISTDiffusion(
                28,
                in_channels,
                timesteps=timesteps,
                time_embedding_dim=time_embedding_dim,
                base_dim=base_dim,
                dim_mults=dim_mults,
            )
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])

        elif args.dataset == "celeba":
            model = CelebADiffusion()

    model = model.to(device)
    model.eval()

    if args.action == "hessian":
        action_func = HessianAction(dt=const_time, xi=path_length, D=1)
    elif args.action == "truncated":
        action_func = TruncatedAction(dt=const_time, xi=path_length)
    elif args.action == "simple":
        action_func = SimpleAction(dt=const_time, gamma=path_length)
    else:
        raise NotImplementedError("Action {args.action} is not implemented")

    # Initialize metrics
    lpips_loss_fn = LPIPS(net="alex").to(device)
    fid_calculator = FrechetInceptionDistance(device=device)
    ppl = 0
    pdv = 0

    iters = 0
    # loop through test dataset
    while init_im is not None and final_im is not None and iters < max_batches:

        iters += 1

        init_im = init_im.to(device)
        final_im = final_im.to(device)

        noise_1 = torch.randn_like(init_im)
        noise_2 = torch.randn_like(final_im)

        # Forward diffusion for both images
        with torch.no_grad():
            heated_init = model.forward_diffusion(
                init_im, t.unsqueeze(-1).repeat(batch_size), noise_1
            )
            heated_final = model.forward_diffusion(
                final_im, t.unsqueeze(-1).repeat(batch_size), noise_1
            )

        # Generate a range of interpolation factors
        alphas = torch.linspace(0, 1, path_length)

        # Create an initial guess for the optimization using spherical or linear interpolation
        print(f"Initial guess using {initial_guess_method} interpolation")
        interpolated_images = torch.stack(
            [
                initial_guess_fn(heated_init.cpu(), heated_final.cpu(), alpha)
                for alpha in alphas
            ],
            axis=1,
        ).to(device)

        # save initial guess
        # choose random batch element to plot
        batch_idx = np.random.randint(0, batch_size)
        # saved = model.sample_from_t(interpolated_images[batch_idx], t).detach().cpu()
        saved = interpolated_images[batch_idx].detach().cpu()
        to_draw = inv_normalizer(torch.clamp(saved, -1, 1))
        final_draw = []
        final_draw.append(to_draw)

        actions = []

        interpolated_images.requires_grad = True
        optimizer = torch.optim.Adam([interpolated_images], lr=lr)
        scheduler = (
            None  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
        )

        pbar = tqdm(range(steps))

        for i in pbar:

            # It helps to anneal diffusion time

            scale_min = 0.1

            scale_factor = max((1 - i / 1000), scale_min)
            diff_time = torch.tensor((int)(999)).to(device)
            # reshape to (batch_size, path_length)
            diff_time = diff_time.repeat(batch_size * path_length).reshape(
                (batch_size, path_length)
            )
            # for all batch, we want to multiply elements by (1 + 0.5 - abs(j-path_length/2)/path_length) for index j in path length
            # this will make time go higher in the middle of the path
            for j in range(path_length):
                diff_time[:, j] = diff_time[:, j] * (
                    scale_factor
                    - ((scale_factor - scale_min) / 0.5)
                    * abs(j - path_length / 2)
                    / path_length
                )

            # convert to int, and clamp 0-999
            diff_time = diff_time.type(torch.int).clamp(0, 999)

            # finally, flatten
            diff_time = diff_time.flatten()

            # diff_time = torch.tensor(100).to(device)

            # approximate gradient for the vector field. Note that for stable vector fields, finding norm minimizers can also be
            # found by simply integrating over the vector field. This is equivalent to instead taking the gradient of ||y - x||_2^2,
            # where y = x + v(x) and is not differentiated with respect to x. In practice, optimized results look about the same,
            # but there is a huge speedup because we don't need to backprop through the model here.
            if args.truncate_v_gradient:
                # forces = v_scale * model.multi_step_denoising_no_noise(
                #     interpolated_images.reshape(-1, C, H, W), diff_time, v_steps
                # )

                with torch.no_grad():
                    target = interpolated_images.reshape(-1, C, H, W) - model.model(
                        interpolated_images.reshape(-1, C, H, W),
                        diff_time,
                    )

                forces = v_scale * (target - interpolated_images.reshape(-1, C, H, W))

            else:
                forces = v_scale * model.model(
                    interpolated_images.reshape(-1, C, H, W),
                    diff_time,
                )

            # scaling factor between predicted noise and force
            # See Slide 31 of https://docs.google.com/presentation/d/1hVOlNwF1ZEeOfgR7IpU7vETmQ9x7z_dLfWuIaLqqe-w/edit?usp=sharing
            # TODO: don't think this is quite right: the energy based model of our data also should have a temperature dependence,
            # so the scaling factor should be more complicated (maybe an alpha term in the numerator or something)
            # Including the scaling factor makes the actions much higher (2-3k) to start out
            # sqrt_one_minus_alpha_cumprod_t = model.sqrt_one_minus_alphas_cumprod.gather(
            #    -1, t.repeat(batch_size * path_length)
            # ).reshape(batch_size * path_length, 1, 1, 1)

            # TODO The prefactor does not seem to be necessary. The minus sign is inimportant for the hessian action as forces are
            # TODO squared. We will investigate further. Commenting out for now.
            # forces = -noise_pred / sqrt_one_minus_alpha_cumprod_t

            # forces = diff_pred - interpolated_images.reshape(-1, C, H, W)

            # compare norms of forces and forces_alt, with decorative text
            forces = forces.reshape(batch_size, path_length, C, H, W)

            # compute the path norm loss on gaussian blurred images, so that the image manifolds look smoother and the optimized paths can still promote domain transformations
            # (note that even very small domain transformations can have high L2 norm, but after gaussian blur these transformations have smaller norm)
            if g_sigma > 0:
                # apply gaussian blur to use blurred norm for path length term
                interpolated_images_blur = transforms.functional.gaussian_blur(
                    interpolated_images.reshape(-1, C, H, W),
                    kernel_size=(9, 9),
                    sigma=(g_sigma, g_sigma),
                ).reshape((batch_size, path_length, C, H, W))

                forces = forces * torch.linalg.norm(
                    torch.flatten(
                        interpolated_images_blur[:, 0, :, :, :]
                        - interpolated_images_blur[:, -1, :, :, :],
                        start_dim=1,
                    ),
                    dim=1,
                ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # compute the OM action from these forces (vmaped over the batch dimension)
                total_action = torch.vmap(action_func)(
                    interpolated_images_blur, forces
                ).sum()

                # print first and second components
                # with torch.no_grad():
                #     first_component = torch.square((interpolated_images_blur[1:] - interpolated_images_blur[:-1]) / const_time).sum()

                #     second_component = torch.square(forces[:-1] / path_length).sum()

                #     print(f'First component: {first_component.item()}, second component: {second_component.item()}')
            else:
                forces = forces * torch.linalg.norm(
                    torch.flatten(
                        interpolated_images[:, 0, :, :, :]
                        - interpolated_images[:, -1, :, :, :],
                        start_dim=1,
                    ),
                    dim=1,
                ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                total_action = torch.vmap(action_func)(
                    interpolated_images, forces
                ).sum()

            actions.append(total_action.unsqueeze(0).cpu().detach())
            pbar.set_description(f"Optimizing OM action: {total_action.item()}")

            optimizer.zero_grad()

            # compute gradient of the action w.r.t. the images
            (grads,) = torch.autograd.grad(total_action, interpolated_images)
            with torch.no_grad():

                # zero out the gradients of the first and last images on the path (they are fixed)
                grads[:, 0], grads[:, -1] = torch.zeros_like(
                    interpolated_images[:, 0]
                ), torch.zeros_like(interpolated_images[:, -1])

                # assign gradients and take a gradient descent step
                interpolated_images.grad = grads
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # add noise to the non-endpoint images (to keep it in the distribution of the model)
                # TODO: what is the exact reasoning behind this? And how is the noise scale determined? Is it just undoing the OM optimization?
                # noise = torch.randn_like(interpolated_images)
                # noise[:, 0], noise[:, -1] = torch.zeros_like(
                #     interpolated_images[:, 0]
                # ), torch.zeros_like(interpolated_images[:, -1])
                # interpolated_images += noise_perturb_scale * noise

                if i % save_every == 0:
                    # save interpolation path
                    # save = model.sample_from_t(interpolated_images[batch_idx], t)
                    save = interpolated_images[batch_idx].detach().clone()
                    save = inv_normalizer(torch.clamp(save, -1.0, 1.0))
                    final_draw.append(save.cpu())

        # free up gpu memory
        # optimizer.zero_grad()
        # del total_action, grads, forces
        # torch.cuda.empty_cache()

        # print out list of force norms throughout the path of final_draw
        # with torch.no_grad():
        #     forces_finpath = model.model(
        #         interpolated_images[batch_idx],
        #         torch.tensor([100]).repeat(path_length).to(device),
        #     )

        #     # output norms of forces
        #     print(
        #         "Norms of forces: ",
        #         torch.linalg.norm(forces_finpath.reshape(path_length, C * H * W), dim=1)
        #         .cpu()
        #         .detach()
        #         .numpy(),
        #     )
        #     # print path norm
        #     if g_sigma > 0:
        #         print(
        #             "Path norm: ",
        #             torch.sqrt(
        #                 torch.square(
        #                     (
        #                         interpolated_images_blur[:, 1:, :, :, :]
        #                         - interpolated_images_blur[:, :-1, :, :, :]
        #                     )
        #                     / const_time
        #                 ).sum(dim=(0, 2, 3, 4))
        #             )
        #             .cpu()
        #             .detach()
        #             .numpy(),
        #         )

        with torch.no_grad():
            # run reverse diffusion on the final, optimized path
            interpolated_images = model.sample_from_t(
                interpolated_images.reshape(-1, C, H, W), t
            )

            interpolated_images = interpolated_images.reshape(
                batch_size, path_length, C, H, W
            )

            clamped_interpolated_images = torch.clamp(interpolated_images, -1.0, 1.0)

            # Repeat grayscale channel 3 times
            if clamped_interpolated_images.shape[-3] == 1:
                clamped_interpolated_images = clamped_interpolated_images.repeat(
                    1, 1, 3, 1, 1
                )
                init_im = init_im.repeat(1, 3, 1, 1)
                final_im = final_im.repeat(1, 3, 1, 1)

            # Unnormalize images back to [0, 1]
            unnormalized_images = inv_normalizer(clamped_interpolated_images)
            init_im = inv_normalizer(init_im)
            final_im = inv_normalizer(final_im)

            # add to final draw
            if t > 0:
                final_draw.append(unnormalized_images[batch_idx].detach().cpu())

            # Calculate PPL/PDV metrics (on [-1, 1] images)
            _ppl, _pdv = perceptual_path_length_and_variance(
                clamped_interpolated_images, lpips_loss_fn
            )
            ppl += _ppl.mean()
            pdv += _pdv.mean()

            # batch the fid update for memory
            def update_fid_in_batches(fid_calculator, images, batch_size, is_initial):
                num_images = images.shape[0]
                for i in range(0, num_images, batch_size):
                    batch_images_fid = images[i : i + batch_size]
                    fid_calculator.update(batch_images_fid, is_initial)

            batch_size_fid = 64
            update_fid_in_batches(
                fid_calculator, torch.cat([init_im, final_im]), batch_size_fid, True
            )
            unnormalized_reshaped_images = unnormalized_images[:, 1:-1].reshape(
                -1, unnormalized_images.shape[-3], H, W
            )
            update_fid_in_batches(
                fid_calculator, unnormalized_reshaped_images, batch_size_fid, False
            )

        # Plot and log actions and images
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_image(
            torch.cat(final_draw),
            f"../mnist_outputs/grid_{now}.png",
            nrow=final_draw[0].shape[0],
        )

        if not args.disable_logging:
            im_dict = {
                f"Decoded Interpolation Path Steps": wandb.Image(
                    f"../mnist_outputs/grid_{now}.png"
                )
            }

            if steps != 0:
                plt.figure()
                plt.plot(np.arange(steps), torch.stack(actions).cpu().detach())
                plt.xlabel("Optimization Steps")
                plt.ylabel("OM Action")
                plt.title("OM Action vs Optimization Steps")
                im_dict.update({"OM Action": wandb.Image(plt)})
                plt.close()

            wandb.log(im_dict, step=iters - 1)

        # go to next pair of images

        init_im, _ = next(test_iterator)
        final_im, _ = next(test_iterator)

    # PPL
    ppl = ppl / iters
    print("Perceptual Path Length (PPL) Score: ", ppl.item())

    # PDV
    pdv = pdv / iters
    print("Perceptual Distance Variance (PDV) Score: ", pdv.item())

    # FID
    fid = fid_calculator.compute()
    print("Frechet Inception Distance (FID) Score: ", fid.item())
    if not args.disable_logging:
        wandb.log({"PPL": ppl.item(), "PDV": pdv.item(), "FID": fid.item()})
        wandb.run.summary.update(
            {"PPL": ppl.item(), "PDV": pdv.item(), "FID": fid.item()}
        )
