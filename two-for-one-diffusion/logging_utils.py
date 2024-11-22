import numpy as np
from PIL import Image
import io
import os
import torch
from tqdm import tqdm
import gsd.hoomd
import numpy as np

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
from IPython.display import Image as IPyImage, display
from utils import center_zero
from rmsd import kabsch_rotate


def save_ovito_traj(
    positions, filename, alpha_carbon_lim=100000, all_backbone=False, align=False
):
    """
    Save the given positions to a GSD file using Ovito.
    """

    t = gsd.hoomd.open(name=filename, mode="w")
    cell = 1.5 * torch.eye(3) * positions.cpu().abs().max()

    if align:
        positions = center_zero(positions)

    if not all_backbone:
        positions = positions[:, :alpha_carbon_lim]
    for i, pos in enumerate(positions):
        if align:
            pos = kabsch_rotate(pos, positions[0])
        t.append(create_frame(i, pos, cell, alpha_carbon_lim, all_backbone))

    t.close()


def create_frame(step, position, cell, alpha_carbon_lim, all_backbone):
    """
    Create an Ovito frame from the given positions.
    """
    # Particle positions, velocities, diameter
    # TODO: add option to add bonds between C and N atoms

    natoms = position.shape[0]
    position = torch.Tensor(position)
    partpos = position.tolist()
    diameter = 0.8 * np.ones((natoms,))
    diameter = diameter.tolist()
    # Now make gsd file
    s = gsd.hoomd.Frame()
    s.configuration.step = step
    s.particles.N = natoms
    s.particles.position = partpos
    s.particles.diameter = diameter
    s.configuration.box = [cell[0][0], cell[1][1], cell[2][2], 0, 0, 0]

    # Bonds for visualization
    # TODO: add option to include bonds between backbone and CB/N atoms
    senders = np.arange(min(position.shape[0], alpha_carbon_lim) - 1)
    receivers = np.arange(1, min(position.shape[0], alpha_carbon_lim))
    if all_backbone:
        # construct bonds between CA and N atoms AND between CA and CB atoms
        N_senders = senders
        N_receivers = np.arange(alpha_carbon_lim, 2 * alpha_carbon_lim - 1)
        CB_senders = senders
        CB_receivers = np.arange(2 * alpha_carbon_lim, 3 * alpha_carbon_lim - 1)

        senders = np.concatenate([senders, N_senders, CB_senders])
        receivers = np.concatenate([receivers, N_receivers, CB_receivers])

    bonds = np.stack([senders, receivers], axis=1)

    s.bonds.N = bonds.shape[0]
    s.bonds.group = bonds
    return s


def visualize_interpolation(
    protein_name, gen_mode, append_exp_name, subsample=None, ref_path=None
):
    """
    Ball-and-stick visualization of the interpolated path.
    """
    # Load data
    append_exp_name_str = "_" + append_exp_name if append_exp_name else ""
    eval_folder = f"../saved_models/{protein_name}/main_eval_output_{gen_mode}{append_exp_name_str}"
    sample_path = Path(eval_folder, f"sample-{gen_mode}.pt")
    pdb_file = (
        f"../datasets/folded_pdbs/{Molecules[protein_name.upper()].value}-0-c-alpha.pdb"
    )

    # Load sampled molecules
    sampled_mol = torch.load(sample_path)
    if subsample is not None:
        sampled_mol = sampled_mol[np.random.permutation(subsample)]
    n_atoms = sampled_mol.shape[1]

    vis_in = sampled_mol.reshape(sampled_mol.shape[0], -1, n_atoms, 3)[0].permute(
        1, 0, 2
    )
    visualization = get_interpolation_viz(vis_in)
    return visualization


def get_interpolation_viz(interpolated_pos):
    """
    Interpolated_pos: [N_atoms, path_length, 3] interpolated positions.
    """
    if not isinstance(interpolated_pos, np.ndarray):
        interpolated_pos = interpolated_pos.cpu().numpy()
    # Create bonds for visualization
    senders = np.arange(interpolated_pos.shape[0] - 1)
    receivers = np.arange(1, interpolated_pos.shape[0])
    bonds = np.stack([senders, receivers], axis=1)

    bonds_seq = np.repeat(bonds[None, ...], interpolated_pos.shape[1], axis=0)
    # visualize all the samples
    node_idx = np.arange(interpolated_pos.shape[0])

    interpolation_vis = visualize_trajectory(interpolated_pos, bonds_seq, node_idx)
    return interpolation_vis


# Convert numpy array into a list of PIL Images
def visualize_gif(frames):
    image_sequence = []
    for frame in frames:
        # Convert RGBA to RGB
        img = Image.fromarray(frame[:3, :, :].transpose(1, 2, 0), "RGB").resize(
            (320, 240)
        )
        image_sequence.append(img)

    # Save as a GIF
    gif_path = "generated_animation.gif"
    image_sequence[0].save(
        gif_path, save_all=True, append_images=image_sequence[1:], duration=100, loop=0
    )
    display(IPyImage(data=open(gif_path, "rb").read()))
    os.remove(gif_path)


def visualize_trajectory(
    pos_seq: np.ndarray,
    bonds_seq: np.ndarray,
    node_idx: np.ndarray,
):
    """
    Visualize a trajectory of atomic graphs in 3D and save as gifs.
    pos_seq: (n_node, n_step, 3)
    bonds_seq: (n_step, n_edge, 2)
    node_idx: (n_node', ), nodes index to visualize

    return numpy array of shape (time, channel, height, width)
    """

    # Get the bonds index
    max_node_idx = node_idx.max()
    min_node_idx = node_idx.min()

    def get_bond_idx(bonds):
        return np.argwhere(
            (bonds[:, 0] >= min_node_idx) & (bonds[:, 0] <= max_node_idx)
        ).squeeze()

    bonds_idx = [get_bond_idx(bonds) for bonds in bonds_seq]

    pos_seq = pos_seq[node_idx, ...]

    # Center position with the first frame
    pos_seq = pos_seq - pos_seq[:, 0:1, :].mean(axis=0, keepdims=True)

    # Center around zero
    pos_seq = pos_seq - (pos_seq.max(axis=(0, 1)) + pos_seq.min(axis=(0, 1))) / 2

    coord_max = pos_seq.max(axis=(0, 1))
    coord_min = pos_seq.min(axis=(0, 1))
    visualization_boundary = coord_max.max()
    coord_max = np.clip(coord_max, -visualization_boundary, visualization_boundary)
    coord_min = np.clip(coord_min, -visualization_boundary, visualization_boundary)
    coord_min = np.nan_to_num(coord_min, nan=-visualization_boundary)
    coord_max = np.nan_to_num(coord_max, nan=visualization_boundary)

    def plot_and_save(pos, bonds, index):
        # A simple and fast visualization for the 3D atomic structure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Draw atoms as spheres
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color="tab:blue", alpha=0.75, s=100)

        # Draw bonds with Line3DCollection
        if bonds.size > 0:
            lines = pos[bonds]
            ax.add_collection3d(
                Line3DCollection(lines, colors="tab:orange", alpha=0.75)
            )

        ax.set_xlim(coord_min[0], coord_max[0])
        ax.set_ylim(coord_min[1], coord_max[1])
        ax.set_zlim(coord_min[2], coord_max[2])

        ax.set_title(f"Step {index}")

        # Save the plot as a numpy array
        with io.BytesIO() as buff:
            fig.savefig(buff, format="raw")
            fig.savefig("test.png")
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1)).transpose(2, 0, 1)

        plt.close(fig)  # Close the figure to free memory
        return im

    plot_and_save_wrapper = lambda index: plot_and_save(
        pos_seq[:, index, ...],
        bonds_seq[index, bonds_idx[index], :] - min_node_idx,
        index,
    )

    # Generate visualizations for each timestep
    image_meta = []
    for i in range(pos_seq.shape[1]):
        image_meta.append(plot_and_save_wrapper(i))

    return np.stack(image_meta, axis=0)
