import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from simpleMB import SimpleMB

import numpy as np
import matplotlib.pyplot as plt


class Args:
    pass


args = Args()
args.device = "cuda"


from tqdm import tqdm
import imageio
from mb_actions import SimpleAction, S2Action
from ase import units

potential = SimpleMB(device="cuda", n_in=2)

x0, xf = potential.initial_point.detach(), potential.final_point.detach()

"""Optimization Hyperparameters"""

line_density = 300  # path length
dt = 5 * units.fs  # time step to use in action
gamma = torch.tensor(0.1 / units.fs)  # gamma value to use in action
D = torch.tensor(500 * units.kB / gamma)  # diffusion coefficient to use in action
# Choose action type
action_cls = SimpleAction
# action_cls = S2Action
action_func = action_cls(potential=potential, dt=dt, gamma=gamma, D=D)

iterations = 1000  # number of optimization steps
alpha = 2e-1  # learning rate
write_every = 100  # how often to write to gif

line_x = torch.linspace(x0[1], xf[1], line_density)
line_y = torch.linspace(x0[0], xf[0], line_density)
line_points = torch.stack((line_x, line_y), axis=-1).to(args.device)

optimizer = torch.optim.Adam([line_points], lr=alpha)

gif_data = []

min_action = 1e9
for i in tqdm(range(iterations)):
    line_points.requires_grad = True
    action = action_func(line_points)
    if action < min_action:
        min_action = action
        best_path = line_points.detach().cpu()

    total_action = action

    optimizer.zero_grad()

    (grads,) = torch.autograd.grad(total_action, line_points)

    with torch.no_grad():
        grads[0, :], grads[-1, :] = torch.zeros(2).to(args.device), torch.zeros(2).to(
            args.device
        )
        line_points.grad = grads
        optimizer.step()

        grads = grads.cpu()

        # Calculate the alternative action
        # s_action = simple_action(line_points)

        draw_points = line_points.detach().cpu()

    if i % write_every == 0:
        # print(f"Total action for the step {i} is {(action).cpu().detach().numpy()}")
        num_points = 100
        x_values = torch.linspace(potential.Lx, potential.Hx, num_points)
        y_values = torch.linspace(potential.Ly, potential.Hy, num_points)

        x, y = torch.meshgrid(x_values.to(args.device), y_values.to(args.device))
        z = potential.U_split(x, y).cpu()

        fig, ax = plt.subplots()
        colorbar = ax.imshow(
            z,
            extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
            vmin=potential.U_min,
            vmax=potential.U_max,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        ax.set(xlabel="x-axis", ylabel="y-axis", title="Contour Plot")
        plt.colorbar(colorbar)

        scatter_plot = ax.scatter(
            draw_points[:, 0], draw_points[:, 1], s=1, c="red", label="Line"
        )
        quiver_plot = ax.quiver(
            draw_points[:, 0],
            draw_points[:, 1],
            -grads[:, 0],
            -grads[:, 1],
            scale_units="xy",
            angles="xy",
            color="blue",
            alpha=0.7,
        )
        ax.legend()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_data.append(image)


from mb_dataset import MBDataset
from mb_calculator import MullerBrownPotential

calculator = MullerBrownPotential(device="cpu")
dataset = MBDataset(
    save_path="data_final",
    transition_path_guess=np.fliplr(draw_points.numpy()),
    timestep=20.0,
    temperature=800,
    n_steps=100000,
    n_sims=100,
    gamma=0.01,
)
# dataset = MBDataset(save_path = 'data', transition_path_guess=None, timestep = 5.0, temperature = 500, n_steps = 100, n_sims = 50)

# dataset = MBDataset(preload_sim_dir="data/temp=1000_timestep=5.0_friction=0.1")

# from torch.utils.data import DataLoader
# mb_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# num_points = 100
# x_values = torch.linspace(calculator.Lx, calculator.Hx , num_points)
# y_values = torch.linspace(calculator.Ly, calculator.Hy, num_points)

# x, y = torch.meshgrid(x_values, y_values)
# z = calculator.U_split(x, y).cpu()

# fig, ax = plt.subplots()
# colorbar = ax.imshow(z, extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), vmin=calculator.U_min, vmax=calculator.U_max, origin='lower', cmap='viridis', aspect='auto')
# ax.set(xlabel="x-axis", ylabel="y-axis", title="Contour Plot")
# plt.colorbar(colorbar)

# # also plot batch from dataloader
# for i, batch in enumerate(mb_dataloader):
#     ax.scatter(batch[:, 1], batch[:, 0], color='red', s=0.1)

# ax.scatter(draw_points[:,0], draw_points[:,1], s=1, c='blue', label='Line')

# ax.set_xlim([calculator.Lx, calculator.Hx])
# ax.set_ylim([calculator.Ly, calculator.Hy])

# plt.show()
# plt.close()
