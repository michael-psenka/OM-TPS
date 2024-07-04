import torch


class SimpleAction(torch.nn.Module):
    """
    Simple Onsager-Machlup action. For details, see Eqn 4 from the paper:
    Lee, J., Lee, IH., Joung, I. et al. Finding multiple reaction pathways via
    global optimization of action. Nat Commun 8, 15443 (2017). https://doi.org/10.1038/ncomms15443

    They provide a simplified Onsager-Machlup action, which avoids computing the Hessian of the energy and is thus more efficient.
    """

    def __init__(self, dt, gamma):
        """
        Args:
            dt: float, time step
            gamma: float, diffusion coefficient
        """
        super(SimpleAction, self).__init__()
        self.dt = dt
        self.gamma = gamma

    def forward(self, path: torch.Tensor, forces: torch.Tensor):
        """
        Args:
            path: torch.Tensor of images of shape [P, C, H, W], where P is the number of images on the path.
            forces: torch.Tensor of forces (derived from diffusion model score estimates) of shape [P, C, H, W], where N is the number of points on the path.
        Returns the OM action of the path (torch.Tensor of shape [1]).

        Note: we omit the term which involves the difference of the energy at the endpoints of the path,
        because it is constant and does not affect the optimization.
        """

        assert path.shape == forces.shape, "path and forces must have the same shape"
        assert len(path.shape) == 4, "path and forces must have shape [P, C, H, W]"

        result = 0.0
        first_term = torch.square(self.gamma / self.dt * (path[1:] - path[:-1]))

        second_term = (torch.square(forces[:-1]) + torch.square(forces[1:])) / 2.0

        third_term = (
            -self.gamma / self.dt * (path[1:] - path[:-1]) * (forces[:-1] - forces[1:])
        )
        result = torch.sum(first_term + second_term + third_term)

        return result * self.dt / (4 * self.gamma)


class HessianAction(torch.nn.Module):
    """
    Onsager-Machlup action involving the Hessian of the energy.
    See Eqn 7 from the paper:
    Artur B. Adib. Stochastic actions for diffusive dynamics: Reweight-
    ing, sampling, and minimization. The Journal of Physical Chemistry B,
    112(19):5910–5916, 05 2008

    Note: we omit the term which involves the difference of the energy at the endpoints of the path,
    because it is constant and does not affect the optimization.
    """

    def __init__(self, dt_xi):
        super(HessianAction, self).__init__()
        self.dt_xi = dt_xi

    def forward(self, path: torch.Tensor, forces: torch.Tensor):
        """
        Args:
            path: torch.Tensor of images of shape [P, C, H, W], where P is the number of images on the path.
            forces: torch.Tensor of diffusion model score estimates of shape [P, C, H, W], where N is the number of points on the path.
        Returns the OM action of the path (torch.Tensor of shape [1]).
        """
        raise NotImplementedError("HessianAction is not implemented yet.")


"""
Slightly different action, TODO: understand the difference
"""

# def simple_action(path, forces):
#     # print(path)

#     result = 0.0
#     for i in range(path.shape[0] - 1):
#         first_term = torch.square((path[i + 1, :] - path[i, :])) * (xi / 4 / dt)
#         f_n = forces[i]
#         f_np = forces[i + 1]
#         second_term = (torch.square(f_n) + torch.square(f_np)) * (dt / 4 / xi)
#         third_term = (
#             (path[i + 1, :] - path[i, :]) * (f_np - f_n) * (dt * D / torch.tensor(2.0))
#         )
#         result = result + torch.sum(first_term + second_term + third_term)

#     return result / torch.tensor(4.0)
