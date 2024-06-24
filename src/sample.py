"""
Produce linear interpolation between two images.
"""

import torch
import torchvision
from torchvision.utils import save_image

from train_mnist import create_mnist_dataloaders


train_dataloader, test_dataloader = create_mnist_dataloaders(1)

train_iterator = iter(train_dataloader)

im_1, target = next(train_iterator)
im_2, target = next(train_iterator)

save_image(im_1,"interpolated/linear/start.png")
save_image(im_2, "interpolated/linear/end.png")

im_1 = (
    torchvision.io.read_image(
        "interpolated/linear/start.png", mode=torchvision.io.ImageReadMode.GRAY
    )
    / 255
)
im_2 = (
    torchvision.io.read_image(
        "interpolated/linear/end.png", mode=torchvision.io.ImageReadMode.GRAY
    )
    / 255
)


# Number of interpolation steps
num_samples = 32

# Generate a range of interpolation factors
alphas = torch.linspace(0, 1, num_samples)
# print(alphas)
# Linearly interpolate between the two images at each alpha
interpolated_images = [torch.lerp(im_1, im_2, alpha) for alpha in alphas]

for i, image in enumerate(interpolated_images):
    save_image(image, "interpolated/linear/interp_{}.png".format(i))
