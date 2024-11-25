"""
Loop over a bunch of different action parameters and log the optimization paths/final actions
Use the same parameters to evaluate the actions so that we have a one-to-one comparison.
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from simpleMB import SimpleMB
from draw_gifs import GifDrawer


from tqdm import tqdm
import imageio
import wandb

from mb_actions import SimpleAction, S2Action, HutchinsonAction, TruncatedAction

# Action parameters

line_density = 80  # number of points on the line
iterations = 1000
alpha = 2e-1
#How many subiterations of Hutch to do
write_every = 100
# WARGNING if use_guess is true, line_density is ignored
use_guess = False
# Save guess for further runs
save_guess = False

global_config = {
    "line_density": line_density,
    "iterations": iterations,
    "alpha": alpha,
    "write_every": write_every,
    "use_guess": use_guess,
    "save_guess": save_guess
}

wandb.login()
wandb.init(
    project="OMBasics", config=global_config, # name = "Define name if desired"
)

os.makedirs("MB_tests", exist_ok=True)

device = (
    torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
)

potential = SimpleMB(device=device, n_in=2)

x0, xf = potential.initial_point.detach(), potential.final_point.detach()

# Gif drawing
gif_drawer = GifDrawer()

gamma = torch.tensor(1.0).to(device)
dt = torch.tensor(1.0).to(device)
D = torch.tensor(1.0)

#Create trunaced action config
trunc_config = {
    "gamma": gamma.item(),
    "dt": dt.item(),
    "D": 0,
    "action": S2Action,
    "name": "Trunc_dt={:.2f}_gamma={:.2f}_D={:.2f}".format(dt.item(), gamma.item(), 0),
} 

s2_config = {
    "gamma": gamma.item(),
    "dt": dt.item(),
    "D": D.item(),
    "action": S2Action,
    "name": "S2dt={:.2f}_gamma={:.2f}_D={:.2f}".format(dt.item(), gamma.item(), D.item()),
} 

hutch_config = {
    "gamma": gamma.item(),
    "dt": dt.item(),
    "D": D.item(),
    "action": HutchinsonAction,
    "name": "Hutch_dt={:.2f}_gamma={:.2f}_D={:.2f}".format(dt.item(), gamma.item(), D.item()),
} 

D = torch.tensor(5.0)
s2_highD = {
    "gamma": gamma.item(),
    "dt": dt.item(),
    "D": D.item(),
    "action": S2Action,
    "name": "S2dt={:.2f}_gamma={:.2f}_D={:.2f}".format(dt.item(), gamma.item(), D.item()),
} 

hutch_highD = {
    "gamma": gamma.item(),
    "dt": dt.item(),
    "D": D.item(),
    "action": HutchinsonAction,
    "name": "Hutch_dt={:.2f}_gamma={:.2f}_D={:.2f}".format(dt.item(), gamma.item(), D.item()),
} 

configs = [trunc_config, s2_config, hutch_config, s2_highD, hutch_highD]

gif_data = []

result_dict = {}
log_dict = {}

if use_guess:
    with open('good_guess.bck', 'rb') as handle:
        good_guess = pickle.load(handle)

print("Will run for {} runs.".format(len(configs)))

for config in configs:

    action_f = config["action"]
    action_func = lambda path: action_f(
            sample_force_func=potential.sample_force_func,
            laplace_func=potential.laplace,
            dt=config["dt"],
            gamma=config["gamma"],
            D=config["D"],
        )(path)
    run_name = config["name"]
    result_dict[run_name] = []

    
    if use_guess:
        line_points = torch.tensor(good_guess).to(device)
    else:
        line_x = torch.linspace(x0[0], xf[0], line_density)
        line_y = torch.linspace(x0[1], xf[1], line_density)
        line_points = torch.stack((line_x, line_y), axis=-1).to(device)
    optimizer = torch.optim.Adam([line_points], lr=alpha)

    print("Running ", run_name)

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
            result_dict[run_name].append((draw_points.detach().cpu().numpy(), grads.detach().cpu().numpy()))

        if i % write_every == 0:
            image = gif_drawer.draw_single_traj(draw_points, grads)
            gif_data.append(image)

    imageio.mimsave(
        f"MB_tests/{run_name}.gif",
        gif_data,
        fps=3,
    )
    
    log_dict[run_name] = wandb.Video(f"MB_tests/{run_name}.gif")

    gif_data = []
    
if save_guess: 
    with open('good_guess.bck', 'wb') as handle:
        pickle.dump(line_points.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save all points for future plotting if needed.
with open('trajectories.bck', 'wb') as handle:
    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

gif_drawer.draw_all_data(result_dict)
log_dict["all_runs"] = wandb.Video(f"MB_tests/all_runs.gif")

wandb.log(log_dict)
wandb.run.summary.update(log_dict)
wandb.finish()

"""

# Plot gif of OM actions
plt.figure()
plt.plot(np.arange(len(configs)), actions)
plt.xlabel("Optimization Steps")
plt.ylabel("OM Action")
plt.title("OM Action vs Optimization Steps")
log_dict.update({"OM Action": wandb.Image(plt)})
plt.save("opt.gif")
plt.close()

eval_action = evaluation_action(line_points) + evaluation_action(
    torch.flip(line_points, dims=(0,))
)
log_dict.update({"final_eval_action": eval_action.item()})
wandb.log(log_dict, step=0)
wandb.run.summary.update(log_dict)
wandb.finish()

"""