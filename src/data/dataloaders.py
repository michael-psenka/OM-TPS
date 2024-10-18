from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, ImageNet
from torchvision import transforms

import os
from PIL import Image


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):

    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = MNIST(
        root="data/mnist", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="data/mnist", train=False, download=True, transform=preprocess
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
        root="data/sanjeevr/imagenet", split="train", transform=preprocess
    )
    test_dataset = ImageNet(
        root="data/sanjeevr/imagenet", split="val", transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all image file paths in the directory
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load the image
        image = Image.open(img_path)

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)

        # return blank label to conform
        return image, 0


def create_celeba_dataloaders(batch_size, root_dir):
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),  # Resize to 256x256 if necessary
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Define the directories
    train_dir = root_dir + "/train"
    valid_dir = root_dir + "/valid"

    # Create the train and validation datasets using the custom dataset
    train_dataset = CelebADataset(train_dir, transform=transform)
    valid_dataset = CelebADataset(valid_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader
