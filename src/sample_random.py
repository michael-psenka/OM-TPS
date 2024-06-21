import torch
import argparse
import torchvision
from torchvision.utils import save_image


from model import MNISTDiffusion

from utils import ExponentialMovingAverage


parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt", type=str, help="define checkpoint path", default="")
parser.add_argument(
    "--n_samples",
    type=int,
    help="define sampling amounts after every epoch trained",
    default=36,
)
parser.add_argument("--model_base_dim", type=int, help="base dim of Unet", default=64)
parser.add_argument(
    "--timesteps", type=int, help="sampling steps of DDPM", default=1000
)
parser.add_argument(
    "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
)
parser.add_argument(
    "--model_ema_decay", type=float, help="ema model decay", default=0.995
)
parser.add_argument(
    "--log_freq", type=int, help="training log message printing frequence", default=10
)
parser.add_argument(
    "--no_clip",
    action="store_true",
    help="set to normal sampling method without clip x_0 which could yield unstable samples",
)
parser.add_argument("--cpu", action="store_true", help="cpu training")

args = parser.parse_args()

device = "cpu" if args.cpu else "cuda"

in_channels = 1
time_embedding_dim = 256
timesteps = 1000
base_dim = 64
dim_mults = [2, 4]

adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
alpha = 1.0 - args.model_ema_decay
alpha = min(1.0, alpha * adjust)

model = MNISTDiffusion(
    28,
    in_channels,
    timesteps=timesteps,
    time_embedding_dim=time_embedding_dim,
    base_dim=base_dim,
    dim_mults=dim_mults,
)
model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)


ckpt = torch.load("results/best_models/best_model.pt")
model_ema.load_state_dict(ckpt["model_ema"])
model.load_state_dict(ckpt["model"])

model = model.to(device)
model_ema.eval()

# images = (model_ema.module.sampling(16) + 1.)/2. #[-1,1] to [0,1]
model.eval()
images = (model.sampling(16) + 1.0) / 2.0  # [-1,1] to [0,1]

print(images.min(), images.max())

save_image(images, "random_samples/random_16.png")
