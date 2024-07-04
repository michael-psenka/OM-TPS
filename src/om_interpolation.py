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
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from train_mnist import create_mnist_dataloaders

from model import MNISTDiffusion
from actions import SimpleAction
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
        default=300,
    )

    parser.add_argument(
        "--initial_guess_method",
        type=str,
        help="method to generate initial interpolation path (options: 'spherical' or 'linear')",
        default="linear",
    )

    parser.add_argument(
        "--path_length", type=int, help="length of interpolation path", default=8
    )
    parser.add_argument(
        "--steps", type=int, help="number of OM optimization steps", default=500
    )
    parser.add_argument(
        "--const_time", type=float, help="constant time for OM action", default=4.0
    )

    parser.add_argument(
        "--noise_perturb_scale",
        type=float,
        help="scale of noise perturbing OM path after every optimization step",
        default=0.06,
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
        "--batch_size",
        type=int,
        help="batch_size for doing the optimization",
        default=192,  # this saturates GPU memory on Sanjeev's Germain server
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu" if args.cpu else "cuda"

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

    t = torch.tensor(latent_time).unsqueeze(-1).to(device)

    os.makedirs("../mnist_outputs/", exist_ok=True)

    # Instantiate dataloader
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=batch_size)
    train_iterator = iter(train_dataloader)
    test_iterator = iter(test_dataloader)

    init_im, _ = next(test_iterator)
    final_im, _ = next(test_iterator)
    _, C, H, W = init_im.shape

    # Un-normalize images from [-1, 1]  to [0, 1]
    inv_normalizer = transforms.Normalize([-1.0], [2.0])

    # Instantiate model
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

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    model = model.to(device)
    model.eval()

    # Define OM Action
    simple_action = SimpleAction(dt=const_time, gamma=path_length)

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
            heated_init = model._forward_diffusion(
                init_im, t.repeat(batch_size), noise_1
            )
            heated_final = model._forward_diffusion(
                final_im, t.repeat(batch_size), noise_1
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
        saved = model.sample_from_t(t, interpolated_images[batch_idx]).detach().cpu()
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

            # compute diffusion model score estimates
            noise_pred = model.model(
                interpolated_images.reshape(-1, C, H, W),
                t.repeat(batch_size * path_length),
            )

            # scaling factor between predicted noise and force
            # See Slide 31 of https://docs.google.com/presentation/d/1hVOlNwF1ZEeOfgR7IpU7vETmQ9x7z_dLfWuIaLqqe-w/edit?usp=sharing
            # TODO: don't think this is quite right: the energy based model of our data also should have a temperature dependence,
            # so the scaling factor should be more complicated (maybe an alpha term in the numerator or something)
            # Including the scaling factor makes the actions much higher (2-3k) to start out
            sqrt_one_minus_alpha_cumprod_t = model.sqrt_one_minus_alphas_cumprod.gather(
                -1, t.repeat(batch_size * path_length)
            ).reshape(batch_size * path_length, 1, 1, 1)

            forces = -noise_pred / sqrt_one_minus_alpha_cumprod_t
            forces = forces.reshape(batch_size, path_length, C, H, W)

            # compute the OM action from these forces (vmaped over the batch dimension)
            total_action = torch.vmap(simple_action)(interpolated_images, forces).mean()
            test_model_grads = torch.autograd.grad(
                total_action,
                model.model.parameters(),
                retain_graph=True,
                allow_unused=True,
            )[0]
            assert all(
                [grad is not None for grad in test_model_grads]
            ), "Action gradient w.r.t model parameters is None. Gradients wont be tracked correctly through the diffusion model."

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
                noise = torch.randn_like(interpolated_images)
                noise[:, 0], noise[:, -1] = torch.zeros_like(
                    interpolated_images[:, 0]
                ), torch.zeros_like(interpolated_images[:, -1])
                interpolated_images += noise_perturb_scale * noise

                if i % save_every == 0:
                    # save interpolation path
                    save = model.sample_from_t(t, interpolated_images[batch_idx])
                    save = torch.clamp(save, -1.0, 1.0)
                    final_draw.append(save.cpu())

        with torch.no_grad():
            # run reverse diffusion on the final, optimized path
            interpolated_images = model.sample_from_t(
                t, interpolated_images.reshape(-1, C, H, W)
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

            # Calculate PPL/PDV metrics (on [-1, 1] images)
            _ppl, _pdv = perceptual_path_length_and_variance(
                clamped_interpolated_images, lpips_loss_fn
            )
            ppl += _ppl.mean()
            pdv += _pdv.mean()

            # Update fid calculator (on [0, 1] images)
            fid_calculator.update(torch.cat([init_im, final_im]), True)
            fid_calculator.update(
                unnormalized_images[:, 1:-1].reshape(
                    -1, unnormalized_images.shape[-3], H, W
                ),
                False,
            )

        # Plot and log actions and images
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_image(
            torch.cat(final_draw),
            f"../mnist_outputs/grid_{now}.png",
            nrow=final_draw[0].shape[0],
        )
        if not args.disable_logging:
            if steps != 0:
                plt.plot(np.arange(steps), torch.stack(actions).cpu().detach())
                plt.xlabel("Optimization Steps")
                plt.ylabel("OM Action")
                plt.title("OM Action vs Optimization Steps")
                wandb.log({"OM Action": wandb.Image(plt)})

            wandb.log(
                {
                    f"Decoded Interpolation Path Steps": wandb.Image(
                        f"../mnist_outputs/grid_{now}.png"
                    )
                }
            )

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
