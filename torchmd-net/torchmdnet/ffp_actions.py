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

    def forward(self, z: torch.Tensor, path: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            z: Atomic numbers
            path: Implicitly batched path of shape [N_atoms*path_length, 2]
            batch: batch indices
        """
        offset = (batch == 0).count_nonzero().item()  # number of atoms in each molecule
        batch_n = batch[batch < batch.max()]
        batch_np = batch[batch > 0]
        z_n = z[batch < batch.max()]
        z_np = z[batch > 0]

        first_term = torch.square((path[offset:] - path[:-offset])) * (
            self.gamma / 4 / self.dt
        )
        second_term = torch.square(self.force_func(z_n, path[:-offset], batch_n)) * (
            self.dt / 4 / self.gamma
        )
        third_term = self.laplace_func(z_n, path[:-offset], batch_n) * (
            self.dt * self.D / torch.tensor(2.0)
        )
        result = torch.sum(first_term + second_term + third_term)
        return result


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

    def forward(self, z: torch.Tensor, path: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            z: Atomic numbers
            path: Implicitly batched path of shape [N_atoms*path_length, 2]
            batch: batch indices
        """
        offset = (batch == 0).count_nonzero().item()  # number of atoms in each molecule
        batch_n = batch[batch < batch.max()]
        z_n = z[batch < batch.max()]

        first_term = torch.square((path[offset:] - path[:-offset])) * (
            self.gamma / 4 / self.dt
        )
        second_term = torch.square(self.force_func(z_n, path[:-offset], batch_n)) * (
            self.dt / 4 / self.gamma
        )
        result = torch.sum(first_term + second_term)
        return result


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

    def forward(self, z: torch.Tensor, path: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            z: Atomic numbers
            path: Implicitly batched path of shape [N_atoms*path_length, 2]
            batch: batch indices
        """

        offset = (batch == 0).count_nonzero().item()  # number of atoms in each molecule
        batch_n = batch[batch < batch.max()]
        batch_np = batch[batch > 0]
        z_n = z[batch < batch.max()]
        z_np = z[batch > 0]
        first_term = torch.square((path[offset:] - path[:-offset])) * (
            self.gamma / self.dt
        )
        f_n = self.force_func(z_n, path[:-offset], batch_n).to(path.device)
        f_np = self.force_func(z_np, path[offset:], batch_np).to(path.device)
        second_term = (torch.square(f_n) + torch.square(f_np)) * (
            self.dt / self.gamma / 2.0
        )
        third_term = (path[offset:] - path[:-offset]) * (f_np - f_n)
        result = torch.sum(first_term + second_term + third_term)
        return result / torch.tensor(4.0)
