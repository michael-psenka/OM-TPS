import torch
from tqdm import tqdm


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
        self,
        _path: torch.Tensor,
        _forces: torch.Tensor = None,
        path_term_only=False,
        force_term_only=False,
    ):
        """
        Args:
            path: tuple consisting of one or more elements of shape [P, N, 3]
            forces: tuple consisting of one or more elements of shape [P, N, 3]

        For Two for One, the path and forces are just tensors (just the alpha-Carbon coordinates and scores)
        For DiG, the path and forces are tuples of length 2 (alpha-Carbon and residue rotation values and scores)

        """
        N_ref = torch.tensor([1.45597958, 0.0, 0.0])
        C_ref = torch.tensor([-0.533655602, 1.42752619, 0.0])
        ref = torch.stack([N_ref, C_ref], dim=0)

        path_term_all = 0
        force_term_all = 0
        if not isinstance(_path, tuple):
            _path = (_path,)
        if _forces is None or None in _forces:
            if not path_term_only:
                _forces = self.force_func(_path)
            else:
                _forces = (None,) * len(_path)

        if not isinstance(_forces, tuple):
            _forces = (_forces,)

        for i, (path, forces) in enumerate(zip(_path, _forces)):
            path_term = torch.zeros_like(path)
            force_term = torch.zeros_like(path)

            if not force_term_only:

                if len(path_term.shape) == 4:
                    path_term = torch.matmul(ref.to(path_term.device), path_term)

                path_term = torch.square((path[1:] - path[:-1])) * (
                    self.gamma / 4 / self.dt
                )
            if not path_term_only:
                force_term = torch.square(forces) * (self.dt / 4 / self.gamma)
            path_term_all += path_term.sum()
            force_term_all += torch.zeros_like(force_term.sum())

        return path_term_all, force_term_all


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
