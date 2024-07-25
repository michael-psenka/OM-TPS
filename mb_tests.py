"""
Loop over a bunch of different action parameters and log the optimization paths/final actions
Use the same parameters to evaluate the actions so that we have a one-to-one comparison.
"""

import torch
import os
from simpleMB import SimpleMB

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import imageio
import wandb

wandb.require("core")

from mb_actions import SimpleAction, S2Action
from src.utils import validate_git_status

validate_git_status()
wandb.login()

os.makedirs("MB_tests", exist_ok=True)

device = (
    torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
)

potential = SimpleMB(device=device, n_in=2)

x0, xf = potential.initial_point.detach(), potential.final_point.detach()


# Action parameters

line_density = 50  # number of points on the line
iterations = 1000
alpha = 2e-1
write_every = 100

# Loop over a bunch of different action parameters

# Evaluate the action with the same hyperparams regardless of the optimization
evaluation_action_cls = SimpleAction(
    force_func=potential.force_func,
    laplace_func=potential.laplace,
    gamma=torch.tensor(0.1).to(device),
    dt=0.1,
    D=torch.tensor(10).to(device),
)
evaluation_action = lambda path: evaluation_action_cls(path)

for action_f in [SimpleAction, S2Action]:
    for gamma in torch.logspace(-2, 1, 5).to(device):
        for dt in torch.logspace(-2, 0, 5):

            if action_f == SimpleAction:
                D_loop = [torch.tensor(0.0).to(device)]
            else:
                D_loop = torch.logspace(-1, 2, 5).to(device)

            for D in D_loop:
                config = {
                    "gamma": gamma.item(),
                    "dt": dt.item(),
                    "D": D.item(),
                    "action": action_f.__name__,
                }
                wandb.init(
                    project="mb-tests-100points-fixedopt", config=config, name="MB_test"
                )

                action_str = "simple" if action_f == SimpleAction else "S2"
                print(
                    f"MB_{action_str}_gamma={round(gamma.item(), 1)}_dt={dt}_D={D.item()}"
                )
                action_func = lambda path: action_f(
                    force_func=potential.force_func,
                    laplace_func=potential.laplace,
                    gamma=gamma,
                    dt=dt,
                    D=D,
                )(path)

                line_x = torch.linspace(x0[0], xf[0], line_density)
                line_y = torch.linspace(x0[1], xf[1], line_density)
                line_points = torch.stack((line_x, line_y), axis=-1).to(device)

                optimizer = torch.optim.Adam([line_points], lr=alpha)

                gif_data = []

                # Choose action

                # plot the potential
                num_points = 100
                x_values = torch.linspace(potential.Lx, potential.Hx, num_points)
                y_values = torch.linspace(potential.Ly, potential.Hy, num_points)

                x, y = torch.meshgrid(x_values, y_values, indexing="xy")
                z = potential.U_split(x.to(device), y.to(device)).cpu()
                actions = []
                for i in tqdm(range(iterations)):
                    line_points.requires_grad = True
                    action = action_func(line_points)
                    reverse_action = action_func(torch.flip(line_points, dims=(0,)))

                    # It seems likely that they are the same. It probably can be proven
                    total_action = action + reverse_action

                    optimizer.zero_grad()

                    (grads,) = torch.autograd.grad(total_action, line_points)

                    with torch.no_grad():
                        grads[0, :], grads[-1, :] = torch.zeros(2), torch.zeros(2)
                        line_points.grad = grads
                        optimizer.step()
                        grads = grads.cpu()
                        draw_points = line_points.detach().cpu()

                    if i % write_every == 0:
                        # print(f"Total action for the step {i} is {(action+reverse_action).detach().numpy()}")
                        fig, ax = plt.subplots()
                        colorbar = ax.imshow(
                            z,
                            extent=(
                                x_values.min(),
                                x_values.max(),
                                y_values.min(),
                                y_values.max(),
                            ),
                            vmin=potential.U_min,
                            vmax=potential.U_max,
                            origin="lower",
                            cmap="viridis",
                            aspect="auto",
                        )
                        ax.set(xlabel="x-axis", ylabel="y-axis", title="Contour Plot")
                        plt.colorbar(colorbar)

                        scatter_plot = ax.scatter(
                            draw_points[:, 0],
                            draw_points[:, 1],
                            s=1,
                            c="red",
                            label="Line",
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

                        # plt.show()

                        # Save the frame to gif_data

                        fig.canvas.draw()
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                        image = image.reshape(
                            fig.canvas.get_width_height()[::-1] + (3,)
                        )
                        gif_data.append(image)
                        actions.append(total_action.item())

                action_str = "simple" if action_f == SimpleAction else "S2"
                imageio.mimsave(
                    f"MB_tests/MB_{action_str}_gamma={round(gamma.item(), 3)}_dt={round(dt.item(), 3)}_D={round(D.item(), 3)}.gif",
                    gif_data,
                    fps=3,
                )
                log_dict = {
                    "gif": wandb.Video(
                        f"MB_tests/MB_{action_str}_gamma={round(gamma.item(), 3)}_dt={round(dt.item(), 3)}_D={round(D.item(), 3)}.gif"
                    )
                }
                plt.close(fig)

                # Plot gif of OM actions
                plt.figure()
                plt.plot(np.arange(len(actions)), actions)
                plt.xlabel("Optimization Steps")
                plt.ylabel("OM Action")
                plt.title("OM Action vs Optimization Steps")
                log_dict.update({"OM Action": wandb.Image(plt)})
                plt.close()

                eval_action = evaluation_action(line_points) + evaluation_action(
                    torch.flip(line_points, dims=(0,))
                )
                log_dict.update({"final_eval_action": eval_action.item()})
                wandb.log(log_dict, step=0)
                wandb.run.summary.update(log_dict)
                wandb.finish()
