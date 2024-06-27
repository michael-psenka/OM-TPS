from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageNet
from torchvision import transforms


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):

    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = MNIST(
        root="/data/sanjeevr/mnist", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="/data/sanjeevr/mnist", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def create_imagenet_dataloaders(batch_size, image_size=256, num_workers=4):

    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = ImageNet(
        root="/data/sanjeevr/imagenet", split="train", transform=preprocess
    )
    test_dataset = ImageNet(
        root="/data/sanjeevr/imagenet", split="val", transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
