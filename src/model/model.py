import torch.nn as nn
import torch
import math
from .unet import Unet
from .unet_celeba import UNet as CelebAUnet
from tqdm import tqdm

from diffusers import DDPMPipeline

class CelebADiffusion(nn.Module):
    def __init__(
        self):
        super().__init__()

        model_id = "google/ddpm-celebahq-256"

        # load model and scheduler
        self.ddpm = DDPMPipeline.from_pretrained(model_id, device_map={"": "cuda:0"}) 
        self.model = lambda x, t: self.ddpm.unet(x, t).sample
        
    def sampling(self, n_samples):
        return self.ddpm(batch_size=n_samples, output_type="np.array")
    
    def sample_from_t(self, start_t, image):

        # set step values
        # self.scheduler.set_timesteps(num_inference_steps)

        # for t in self.progress_bar(self.scheduler.timesteps):
        #     # 1. predict noise model_output
        #     model_output = self.unet(image, t).sample

        #     # 2. compute previous image: x_t -> x_t-1
        #     image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1)

        return -1


# essentially a wrapper for the base Unet model, but adding sampling methods in same format
# class CelebADiffusion(nn.Module):
#     def __init__(
#         self, n_channels=3, t_emb_dim=128, timesteps=1000, bilinear=True
#     ):
#         super().__init__()

#         self.model = CelebAUnet(n_channels, t_emb_dim, bilinear=bilinear)
#         self.data_shape = (3,256,256)
#         self.timesteps = timesteps
        
#         self.beta_0 = 0.0001
#         self.beta_T = 0.02
        
#     def forward(self, x, t):
#         return self.model((x,t,))

#     def load_state_dict(self, state_dict):
#         self.model.load_state_dict(state_dict)
    
#     @torch.no_grad()
#     def sampling(self, n_samples, device="cuda"):
#         """
#         Perform the complete sampling step according to p(x_0|x_T)
#         """
#         T = self.timesteps
#         Beta = torch.linspace(self.beta_0, self.beta_T, T).cuda()
#         Alpha = 1 - Beta
#         Alpha_bar = torch.ones(T).cuda()
#         Beta_tilde = Beta + 0
#         for t in range(T):
#             Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
#             if t > 0:
#                 Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
#         Sigma = torch.sqrt(Beta_tilde)

#         x = torch.normal(0, 1, size=(n_samples,) + self.data_shape, device=device)
#         for t in range(T-1,-1,-1):
#             if t % 100 == 0:
#                 print('reverse step:', t)
#             ts = (t * torch.ones((n_samples, 1))).cuda()
#             epsilon_theta = self.model((x,ts,))
#             x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
#             if t > 0:
#                 x = x + Sigma[t] * torch.normal(0, 1, size=(n_samples,) + self.data_shape, device=device)
#         return x
    
#     @torch.no_grad()
#     def sample_from_t(
#         self, start_t, x, device="cuda",
#         addnoise = True, save_intermediate=False):

#         Beta = torch.linspace(self.beta_0, (start_t / self.timesteps)*self.beta_T, start_t).cuda()
#         Alpha = 1 - Beta
#         Alpha_bar = torch.ones(start_t).cuda()
#         Beta_tilde = Beta + 0
#         for t in range(start_t):
#             Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
#             if t > 0:
#                 Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
#         Sigma = torch.sqrt(Beta_tilde)

#         for t in range(start_t-1,-1,-1):
#             if t % 100 == 0:
#                 print('reverse step:', t)
#             ts = (t * torch.ones((x.shape[0], 1))).cuda()
#             epsilon_theta = self.model((x,ts,))
#             x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
#             if t > 0:
#                 x = x + Sigma[t] * torch.normal(0, 1, x.shape, device=device)
#         return x




class MNISTDiffusion(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=32,
        dim_mults=[1, 2, 4, 8],
        use_alt_timesampling=False
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.use_alt_timesampling = use_alt_timesampling

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = Unet(
            timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults
        )

    def forward(self, x, noise):
        # x:NCHW
        if self.use_alt_timesampling:
            t_mask = torch.rand(x.shape[0], device=x.device) < 0.5
            # first dist, just in first 10th
            t = torch.randint(0, self.timesteps // 10, (x.shape[0],)).to(x.device)
            # with probability 1/2, sample from rest
            t_g = torch.randint(self.timesteps // 10, self.timesteps, (x.shape[0],)).to(x.device)
            t[t_mask] = t_g[t_mask]
        else:
            t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        x_T = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size)
        ).to(device)

        x_0 = self.sample_from_t(
            self.timesteps,
            x_T,
            clipped_reverse_diffusion=clipped_reverse_diffusion,
            device=device,
        )

        return x_0

    @torch.no_grad()
    def sample_from_t(
        self, start_t, x_t, clipped_reverse_diffusion=True, device="cuda", display=False,
        addnoise = True, save_intermediate=False):

        if save_intermediate:
            x_t_list = []
        if display:
            ddpm_range = tqdm(range(start_t - 1, -1, -1))
        else:
            ddpm_range = range(start_t - 1, -1, -1)
        for i in ddpm_range:
            if addnoise:
                noise = torch.randn_like(x_t).to(device)
            else:
                noise = torch.zeros_like(x_t).to(device)
            t = torch.tensor([i for _ in range(x_t.shape[0])]).to(device)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
            if save_intermediate:
                x_t_list.append(x_t)

        # Using transformation to do that
        # x_t=(x_t+1.)/2. #[-1,1] to [0,1]
        if save_intermediate:

            # flip the indexing of x_t_list
            x_t_list = x_t_list[::-1]
            return x_t_list
        else:
            return x_t

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        # q(x_{t}|x_{t-1})
        return (
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0
            + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(
                x_0.shape[0], 1, 1, 1
            )
            * noise
        )

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        """
        Classic DDPM sampling.
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        """
        pred = self.model(x_t, t)

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        """
        Classic DDPM sampling.
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        """
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = (
            torch.sqrt(1.0 / alpha_t_cumprod) * x_t
            - torch.sqrt(1.0 / alpha_t_cumprod - 1.0) * pred
        )
        x_0_pred.clamp_(-1.0, 1.0)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            mean = (
                beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            ) * x_0_pred + (
                (1.0 - alpha_t_cumprod_prev)
                * torch.sqrt(alpha_t)
                / (1.0 - alpha_t_cumprod)
            ) * x_t

            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            mean = (
                beta_t / (1.0 - alpha_t_cumprod)
            ) * x_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0

        return mean + std * noise

    # take multiple "gradient steps" with respect to the denoising mean model
    def multi_step_denoising_no_noise(self, x_t, t, num_steps=1, device="cuda"):
        with torch.no_grad():
            x_next = x_t.clone()
            for step in range(num_steps):
                # current t is max(0, t - step)
                curr_t = torch.max(t - step, torch.tensor(0).to(device))
                x_next = x_next - self.model(
                    x_next, curr_t
                )

        # scale down output such that scale of output step is approximately
        # invariant to num_steps. 1/num_steps is too steep since it assumes
        # all gradients are in line and don't shrink, both of which are typically
        # false. 1/sqrt empirically seems to keep the scale invariant, at least
        # as tested in smaller settings. TODO: more testing and rigorous analysis
        return (1 / math.sqrt(num_steps))*x_next - x_t
