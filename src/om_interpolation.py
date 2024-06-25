"""
Interpolation via Onsager-Machlup action minimization in raw image space. 
This script loads a pretrained MNIST diffusion model and uses it to interpolate between two images in the latent space.
We start with a linear interpolation path between the two images.
Then, we optimize the path by minimizing the Onsager-Machlup action.
TODO: consolidate this with the om_interpolation_latent.py script (can set reverse diffusion steps accordingly).
"""

import math
import os

import torch
import torchvision
from torchvision.utils import save_image

from unet import Unet
from model import MNISTDiffusion

from train_mnist import create_mnist_dataloaders

os.makedirs("interpolated/OM2", exist_ok=True)
os.makedirs("interpolated/linear", exist_ok=True)

sampling = 32
images = []
for i in range(sampling):
    images.append(
        torchvision.io.read_image(
            "interpolated/linear/interp_{}.png".format(i),
            mode=torchvision.io.ImageReadMode.GRAY,
        )
        / 255
    )

images_tensor = torch.stack(images, axis=0) * 2 - 1
print(images_tensor.shape)
print(images_tensor.min(), images_tensor.max())

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
# print(model.model(images_tensor))

# TODO Dummy values. This oviously cannot work
# TODO change this to actual diffusion model stuff
xi = 0.1
dt = 0.01  # apparently a very important parameter that requires scientific reasoning.


def simple_action(path, forces):

    result = 0.0
    for i in range(path.shape[0] - 1):
        first_term = torch.square((path[i + 1, :] - path[i, :])) * (xi / dt)
        f_n = forces[i]
        f_np = forces[i + 1]
        second_term = (torch.square(f_n) + torch.square(f_np)) * (dt / xi / 2.0)
        third_term = (path[i + 1, :] - path[i, :]) * (f_np - f_n)
        result = result + torch.sum(first_term + second_term + third_term)

    return result / torch.tensor(4.0)


alpha = 1e-1
images_tensor.requires_grad = True
optimizer = torch.optim.Adam([images_tensor], lr=alpha)

saved = images_tensor.detach().clone()
to_draw = torch.clamp((saved + 1.0) / 2.0, -1, 1)
save_image(
    images_tensor, "interpolated/OM2/initial.png", format="png", nrow=int(math.sqrt(20))
)

steps = 1000
save_every = 100

for i in range(steps):

    print(i, images_tensor.min(), images_tensor.max())

    forces = model.model(images_tensor)

    total_action = simple_action(images_tensor, forces)

    optimizer.zero_grad()

    (grads,) = torch.autograd.grad(total_action, images_tensor)
    # print(torch.max(grads))

    with torch.no_grad():

        grads[0], grads[-1] = torch.zeros_like(images_tensor[0]), torch.zeros_like(
            images_tensor[-1]
        )

        images_tensor.grad = grads
        optimizer.step()

        images_tensor.clamp_(-1, 1)

    if i % save_every == 0:
        to_draw = torch.clamp(((torch.cat((saved, images_tensor))) + 1.0) / 2.0, -1, 1)
        save_image(
            to_draw,
            "interpolated/OM2/steps_{}.png".format(i),
            format="png",
            nrow=int(math.sqrt(20)),
        )


to_draw = torch.clamp(((torch.cat((saved, images_tensor))) + 1.0) / 2.0, -1, 1)
save_image(
    to_draw, "interpolated/OM2/steps_{}.png".format(steps), nrow=int(math.sqrt(20))
)
