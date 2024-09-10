import numpy as np
from PIL import Image
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from IPython.display import Image as IPyImage, display


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


def visualize_contact_map(
    true_positions, predicted_positions: np.ndarray, threshold: float = 10.0
):
    """
    Compute and plot normalized contact maps from atomic positions.

    Args:
        true_positions: numpy array of shape (N_samples, N_residues, 3)
        predicted_positions: numpy array of shape (N_samples, N_residues, 3)
        threshold: float, distance threshold in Angstroms for contact definition

    Returns:
        numpy array of the plot of shape (height, width, channel)
    """

    def compute_normalized_log_contact_map(positions):
        diff = positions[:, :, None, :] - positions[:, None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        contact_map = (distances < threshold).astype(float)
        normalized_log_contact_map = np.log10(
            np.mean(contact_map, axis=0) + 1e-10
        )  # Add small value to avoid log(0)
        return normalized_log_contact_map

    # Compute contact maps
    true_contact_map = compute_normalized_log_contact_map(true_positions[::100])
    predicted_contact_map = compute_normalized_log_contact_map(predicted_positions)

    # Plotting the contact maps side by side with a common colorbar and no axis labels or ticks
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(true_contact_map, cmap="viridis_r")
    axes[0].set_title("True Log Contact Map")
    axes[0].axis("off")  # Remove axis labels and ticks

    im2 = axes[1].imshow(predicted_contact_map, cmap="viridis_r")
    axes[1].set_title("Predicted Log Contact Map")
    axes[1].axis("off")  # Remove axis labels and ticks

    # Add a single colorbar for both images
    fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.1)

    # Save to buffer and convert to numpy array
    with io.BytesIO() as buff:
        fig.savefig(buff, format="png")
        buff.seek(0)
        im = Image.open(buff)

        # Convert image to RGB if it has an alpha channel
        if im.mode == "RGBA":
            im = im.convert("RGB")

    w, h = fig.canvas.get_width_height()
    im_array = np.array(im).reshape((h, w, -1))

    plt.close(fig)  # Close the figure to free memory
    return im_array


# TODO: incorporate k3d at some point
# def make_k3d_plot(positions, positions_estimated, graph, animation_time):
#     n_nodes, n_time, _ = positions.shape
#     color_palette = sns.color_palette("husl", n_nodes)

#     all_positions = np.concatenate([positions, positions_estimated], axis=0)

#     positions_time = {}
#     for time_index in range(n_time):
#         animation_time_index = time_index * animation_time / n_time
#         positions_time[f"{animation_time_index:.2f}"] = all_positions[:, time_index, :]

#     colors = np.array([[DARK_GREEN_HEX] * n_nodes + [0x990000] * n_nodes])

#     plot = k3d.plot(grid_visible=False, camera_auto_fit=True)

#     points = k3d.points(all_positions[:, 0, 0:3], point_size=0.1, colors=colors)
#     plot += points

#     for i in range(n_nodes):
#         plot += k3d.line(
#             np.array(positions[i, :, :]).astype(np.float32),
#             color=convert_rgb_to_hex((int(e * 255) for e in color_palette[i])),
#         )

#     for i in range(n_nodes):
#         plot += k3d.line(
#             np.array(positions_estimated[i, :, :]).astype(np.float32),
#             color=convert_rgb_to_hex((int(e * 255) for e in color_palette[i])),
#             opacity=0.2,
#         )

#     for s, r in zip(graph.senders, graph.receivers):
#         plot += k3d.line(
#             np.array([positions[s, 0, :], positions[r, 0, :]]).astype(np.float32),
#             color=DARK_GREY_HEX,
#         )

#     points.positions = positions_time

#     with open("k3d.html", "w") as f:
#         f.write(plot.get_snapshot())

#     wandb.log({"k3d_visualization": wandb.Html(open("k3d.html"), inject=False)})
