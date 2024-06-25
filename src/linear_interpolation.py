"""
Linear interpolation in the latent space of a diffusion model. 
This script loads a pretrained MNIST diffusion model and uses it to interpolate between two images in the latent space.
We do forward diffusion for both images and then linearly interpolate between the two latent representations.
Then, we sample from the model via reverse diffusion at each interpolated latent representation to get the corresponding image.
"""

import math
import argparse

import torch
import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from model import MNISTDiffusion

parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
parser.add_argument("--cpu", action="store_true", help="cpu training")
args = parser.parse_args()

device = "cpu" if args.cpu else "cuda"


init_im = (
    torchvision.io.read_image(
        "interpolated/linear/start.png", mode=torchvision.io.ImageReadMode.GRAY
    )
    / 255
)
final_im = (
    torchvision.io.read_image(
        "interpolated/linear/end.png", mode=torchvision.io.ImageReadMode.GRAY
    )
    / 255
)


normalizer = transforms.Normalize([0.5], [0.5])  # [0,1] to [-1,1]
inv_normalizer = transforms.Normalize([-1.0], [2.0])

init_im = normalizer(init_im).to(device)
final_im = normalizer(final_im).to(device)

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

ckpt = torch.load("best_model.pt")
model.load_state_dict(ckpt["model"])
model.eval()

model = model.to(device)

noise_1 = torch.randn_like(init_im)
noise_2 = torch.randn_like(final_im)

print(init_im.device)
print(model.alphas.device)

# How far?
t = torch.tensor(400).to(device)
heated_init = model._forward_diffusion(init_im, t, noise_1)
heated_final = model._forward_diffusion(final_im, t, noise_2)

num_samples = 16

# Generate a range of interpolation factors
alphas = torch.linspace(0, 1, num_samples)
# Linearly interpolate between the two images at each alpha

interpolated_images = torch.cat(
    [torch.lerp(heated_init.cpu(), heated_final.cpu(), alpha) for alpha in alphas],
    axis=0,
)

interpolated_images = model.sample_from_t(t, interpolated_images.to(device))
print(interpolated_images)
save_image(inv_normalizer(heated_init), "interpolated/classic/example.png")
save_image(
    inv_normalizer(interpolated_images),
    "interpolated/classic/classic_result{}.png".format(t),
    nrow=int(math.sqrt(20)),
)
