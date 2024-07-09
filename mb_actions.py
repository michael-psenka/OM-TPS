import torch


class S2Action(torch.nn.Module):
    """Action with Hessian"""

    def __init__(self, potential, dt, gamma, D):
        """
        Args:
            potential: Potential function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(S2Action, self).__init__()
        self.potential = potential
        self.dt = dt
        self.gamma = gamma
        self.D = D

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [P, 2]
        """
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / 4 / self.dt)
        second_term = torch.square(self.potential.force_func(path[:-1])[1]) * (
            self.dt / 4 / self.gamma
        )
        third_term = self.potential.laplace(path[:-1]) * (
            self.dt * self.D / torch.tensor(2.0)
        )
        result = torch.sum(first_term + second_term + third_term)
        return result


class SimpleAction(torch.nn.Module):
    """
    Action without hessian.
    Basically same as src.actions.SimpleAction but for 2D MB, not images.
    """

    def __init__(self, potential, dt, gamma, D=None):
        """
        Args:
            potential: Potential function
            dt: float, time step
            gamma: float, diffusion coefficient
        """
        super(SimpleAction, self).__init__()
        self.potential = potential
        self.dt = dt
        self.gamma = gamma

    def forward(self, path: torch.Tensor):
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / self.dt)
        f_n = self.potential.force_func(path[:-1])[1]
        f_np = self.potential.force_func(path[1:])[1]
        second_term = (torch.square(f_n) + torch.square(f_np)) * (
            self.dt / self.gamma / 2.0
        )
        third_term = (path[1:] - path[:-1]) * (f_np - f_n)
        result = torch.sum(first_term + second_term + third_term)
        return result / torch.tensor(4.0)
