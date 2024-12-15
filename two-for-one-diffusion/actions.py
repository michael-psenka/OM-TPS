import torch


class S2Action(torch.nn.Module):
    """Action with Hessian"""

    def __init__(self, force_func, laplace_func, dt, gamma, D):
        """
        Args:
            force_func: Force function
            laplace_func: Laplace function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(S2Action, self).__init__()
        self.force_func = force_func
        self.laplace_func = laplace_func
        self.dt = dt
        self.gamma = gamma
        self.D = D

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [P, N, 3]
        """
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / 4 / self.dt)
        second_term = torch.square(self.force_func(path[:-1])) * (
            self.dt / 4 / self.gamma
        )
        third_term = self.laplace_func(path[:-1]) * (
            self.dt * self.D / torch.tensor(2.0)
        )
        result = torch.sum(first_term + second_term + third_term)
        # return result
        return first_term.sum(), (second_term + third_term).sum()


class TruncatedAction(torch.nn.Module):
    """S2Action with Hessian term ignored."""

    def __init__(self, force_func, dt, gamma, laplace_func=None, D=None):
        """
        Args:
            force_func: Force function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(TruncatedAction, self).__init__()
        self.force_func = force_func
        self.dt = dt
        self.gamma = gamma

    def forward(
        self, path: torch.Tensor, forces: torch.Tensor = None, chunks_of_two=False
    ):
        """
        Args: path of shape [P, *], forces of shape [P, *]
        """
        if chunks_of_two:
            # this is necessary if we subsampled points in the path
            # need to make sure that path terms are computed only on adjacent points
            if len(path.shape) == 2:
                path = path.reshape(-1, 2, path.shape[-1])
            else:
                path = path.reshape(-1, 2, path.shape[-2], path.shape[-1])
            first_term = torch.square((path[:, 1] - path[:, 0])) * (
                self.gamma / 4 / self.dt
            )
        else:
            first_term = torch.square((path[1:] - path[:-1])) * (
                self.gamma / 4 / self.dt
            )
        if forces is not None:
            f_n = forces[:-1]
        else:
            f_n = self.force_func(path[:-1])
        second_term = torch.square(f_n) * (self.dt / 4 / self.gamma)

        return first_term.sum(), second_term.sum()


class SimpleAction(torch.nn.Module):
    """
    Action without hessian.
    Basically same as src.actions.SimpleAction but for 2D MB, not images.
    """

    def __init__(self, force_func, dt, gamma, laplace_func=None, D=None):
        """
        Args:
            force_func: Force function
            dt: float, time step
            gamma: float, diffusion coefficient
        """
        super(SimpleAction, self).__init__()
        self.force_func = force_func
        self.dt = dt
        self.gamma = gamma

    def forward(self, path: torch.Tensor):
        # Args: path of shape [P, N, 3]
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / self.dt)
        f_n = self.force_func(path[:-1]).to(path.device)
        f_np = self.force_func(path[1:]).to(path.device)
        # print("force norm: ", f_n.norm(dim = -1).mean())
        second_term = (torch.square(f_n) + torch.square(f_np)) * (
            self.dt / self.gamma / 2.0
        )
        third_term = (path[1:] - path[:-1]) * (f_np - f_n)
        result = torch.sum(first_term + second_term + third_term)
        return result / torch.tensor(4.0)
