import math

import torch
import torchvision
from torchvision.utils import save_image

from torchvision import transforms

import argparse

from model import MNISTDiffusion

from tqdm import tqdm

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

ckpt = torch.load("results/best_models/best_model.pt")
model.load_state_dict(ckpt["model"])

model = model.to(device)
model.eval()

noise_1 = torch.randn_like(init_im)
noise_2 = torch.randn_like(final_im)

print(init_im.device)
print(model.alphas.device)

# How far?
num_samples = 8
timesteps = 300
t = torch.tensor(timesteps).to(device)
heated_init = model._forward_diffusion(init_im, t, noise_1)
heated_final = model._forward_diffusion(final_im, t, noise_1)

# Generate a range of interpolation factors
alphas = torch.linspace(0, 1, num_samples)
# Linearly interpolate between the two images at each alpha

interpolated_images = torch.cat(
    [torch.lerp(heated_init.cpu(), heated_final.cpu(), alpha) for alpha in alphas],
    axis=0,
)

const_time = 4.0

dt_xi = const_time / num_samples


def simple_action(path, forces):
    # print(path)

    result = 0.0
    for i in range(path.shape[0] - 1):
        first_term = torch.square((path[i + 1, :] - path[i, :])) * (1 / dt_xi)
        f_n = forces[i]
        f_np = forces[i + 1]
        second_term = (torch.square(f_n) + torch.square(f_np)) * (dt_xi / 2.0)
        third_term = (path[i + 1, :] - path[i, :]) * (f_np - f_n)
        result = result + torch.sum(first_term + second_term + third_term)

    return result / torch.tensor(4.0)


alpha = 1e-2
interpolated_images = interpolated_images.to(device)
interpolated_images.requires_grad = True
optimizer = torch.optim.Adam([interpolated_images], lr=alpha)

saved = interpolated_images.detach().clone()
to_draw = torch.clamp((saved + 1.0) / 2.0, -1, 1)
save_image(
    inv_normalizer(interpolated_images),
    "interpolated/enhanced/OMinitial.png",
    format="png",
    nrow=int(math.sqrt(20)),
)

steps = 300
save_every = 100

print(interpolated_images.shape)

for i in tqdm(range(steps), desc="applying OM principle"):

    forces = model.model(interpolated_images, t)

    total_action = simple_action(interpolated_images, forces)

    optimizer.zero_grad()

    (grads,) = torch.autograd.grad(total_action, interpolated_images)

    with torch.no_grad():

        grads[0], grads[-1] = torch.zeros_like(
            interpolated_images[0]
        ), torch.zeros_like(interpolated_images[-1])

        interpolated_images.grad = grads
        optimizer.step()

        noise = torch.randn_like(interpolated_images)
        noise[0], noise[-1] = torch.zeros_like(
            interpolated_images[0]
        ), torch.zeros_like(interpolated_images[-1])
        interpolated_images += 0.06 * noise

        # interpolated_images.clamp_(-1,1)

    if i % save_every == 0:
        clamp = True
        if clamp:
            to_draw = (
                torch.clamp(torch.cat((saved, interpolated_images)), -1.0, 1.0) + 1.0
            ) / 2.0
        else:
            to_draw = torch.clamp(
                (torch.cat((saved, interpolated_images)) + 1.0) / 2.0, 0.0, 1.0
            )
        save_image(
            to_draw,
            "interpolated/enhanced/OMsteps_{}.png".format(i),
            format="png",
            nrow=int(math.sqrt(20)),
        )
        print(to_draw.max(), to_draw.min())


# interpolated_images = interpolated_images.detach().clamp(-1,1)
interpolated_images = model.sample_from_t(t, interpolated_images.to(device))

save_image(inv_normalizer(heated_init), "interpolated/enhanced/example.png")
save_image(
    inv_normalizer(torch.clamp(interpolated_images, -1.0, 1.0)),
    "interpolated/enhanced/enhanced_result{}.png".format(t),
    nrow=int(math.sqrt(20)),
)
