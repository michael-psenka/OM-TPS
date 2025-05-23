import numpy as np
import torch
import imageio

from simpleMB import SimpleMB

import matplotlib.pyplot as plt


class GifDrawer:

    def __init__(self, write_every=100):

        self.colors = ["red", "blue", "black", "pink", "cyan"]

        device = "cpu"
        self.potential = SimpleMB(device=device, n_in=2)
        self.write_every = write_every

        # plot the potential
        num_points = 100
        self.x_values = torch.linspace(self.potential.Lx, self.potential.Hx, num_points)
        self.y_values = torch.linspace(self.potential.Ly, self.potential.Hy, num_points)

        x, y = torch.meshgrid(self.x_values, self.y_values, indexing="xy")
        self.z = self.potential.U_split(x.to(device), y.to(device)).cpu()

    def draw_image(
        self,
        draw_points: np.array,
        grads: np.array,
        name,
        color,
        canvas_tuple,
        quiver_draw=False,
    ):
        # print(f"Total action for the step {i} is {(action+reverse_action).detach().numpy()}")

        fig, ax = canvas_tuple

        ax.set(xlabel="x-axis", ylabel="y-axis", title="Contour Plot")

        scatter_plot = ax.scatter(
            draw_points[:, 0],
            draw_points[:, 1],
            s=3,
            c=color,
            label=name,
        )
        if quiver_draw:
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

    def draw_all_data(self, all_data: np.array):

        images = []

        names = list(all_data.keys())  # len M
        data = [d[:: self.write_every] for d in all_data.values()]  # len M x N

        # N is the number of datapoints in a series
        N = len(data[0])
        # M is a number of series
        M = len(data)

        for i in range(N):
            fig, ax = plt.subplots()

            colorbar = ax.imshow(
                self.z,
                extent=(
                    self.x_values.min(),
                    self.x_values.max(),
                    self.y_values.min(),
                    self.y_values.max(),
                ),
                vmin=self.potential.U_min,
                vmax=self.potential.U_max,
                origin="lower",
                cmap="viridis",
                aspect="auto",
            )
            plt.colorbar(colorbar)

            for j in range(M):
                name = names[j]
                color = self.colors[j]
                points, grads = data[j][i]
                self.draw_image(points, grads, name, color, (plt, ax))
            fig.canvas.draw()

            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close(fig)

        imageio.mimsave(
            f"MB_tests/all_runs.gif",
            images,
            fps=3,
        )
        return images

    def draw_single_traj(self, draw_points, grads):

        # print(f"Total action for the step {i} is {(action+reverse_action).detach().numpy()}")
        fig, ax = plt.subplots()
        colorbar = ax.imshow(
            self.z,
            extent=(
                self.x_values.min(),
                self.x_values.max(),
                self.y_values.min(),
                self.y_values.max(),
            ),
            vmin=self.potential.U_min,
            vmax=self.potential.U_max,
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
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image
