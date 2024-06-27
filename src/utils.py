import torch
from PIL import Image
import time
import gc
from git import Repo
import numpy as np
from IPython import display as IPdisplay


def validate_git_status():
    """
    Check if the git repository is clean to run experiments.
    """
    repo = Repo(".", search_parent_directories=True)
    repo_is_dirty = repo.is_dirty()
    
    assert (
        not repo_is_dirty
    ), "Git repository is dirty! Please commit your changes before running wandb online experiments."
    


def get_initial_guess_fn(initial_guess_method):
    if initial_guess_method == "spherical":
        return slerp
    elif initial_guess_method == "linear":
        return torch.lerp
    else:
        raise ValueError("initial_guess_method must be 'spherical' or 'linear'")


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation between two vectors.
    Source: Andrej Karpathy's gist: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    Args:
        v0 (torch.Tensor): the first vector
        v1 (torch.Tensor): the second vector
        t (torch.Tensor): scalar from 0 to 1 parameterizing the interpolation
    """
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1  # simple linear interpolation
    else:
        theta_0 = np.arccos(dot)  # angle between latent vectors
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2


# torchvision ema implementation
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def display_images(images, save_path):
    """The `display_images` function converts a list of
    image arrays into a GIF, saves it to a specified path
    and returns the GIF object for display. It names the GIF
    file using the current time and handles any errors by printing them out."""

    try:
        # Convert each image in the 'images' list from an array to an Image object.
        images = [
            Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images
        ]

        # Generate a file name based on the current time, replacing colons with hyphens
        # to ensure the filename is valid for file systems that don't allow colons.
        filename = time.strftime("%H:%M:%S", time.localtime()).replace(":", "-")
        # Save the first image in the list as a GIF file at the 'save_path' location.
        # The rest of the images in the list are added as subsequent frames to the GIF.
        # The GIF will play each frame for 100 milliseconds and will loop indefinitely.
        images[0].save(
            f"{save_path}/{filename}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
    except Exception as e:
        # If there is an error during the process, print the exception message.
        print(e)

    # Return the saved GIF as an IPython display object so it can be displayed in a notebook.
    return IPdisplay.Image(f"{save_path}/{filename}.gif")


def print_active_torch_tensors():
    """
    Utility debugging function to print the number of active torch tensors in memory.
    """
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # print(type(obj), obj.size())
                count += 1
                del obj
        except:
            pass
    print(f"{count} tensors in memory")
