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
        self,
        path: torch.Tensor,
        forces: torch.Tensor = None,
        chunks_of_two=False,
        subsample_dimensions_percent=None,
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

        return first_term.sum(), second_term.sum(), torch.tensor(0).to(torch.float32)


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


class HutchinsonAction(torch.nn.Module):
    """
    Action that is the same as S2 Action
    but calculates laplacian using Hutchinson trick using
    random vector of -1s and 1s.
    """

    def __init__(self, force_func, dt, gamma, D, N=1):
        super(HutchinsonAction, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.N = N
        self.force_func = force_func

        def force_and_laplace(
            x: torch.tensor, forces: torch.tensor = None, subsample_dimensions=None
        ):
            result = 0

            
            forces = self.force_func(x)


            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                v = torch.randint(0, 2, forces.shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                # Calculate matrix vector product of Hessian and random vector
                (Av,) = torch.autograd.grad(
                    forces, x, grad_outputs=v, retain_graph=True, create_graph=True
                )
                # Make it a scalar. Minus is because we want the energy Hessian, which is the negative force Jacobian.
                result += -torch.sum(v * Av)

            return forces, result / N

        self.force_and_laplace = force_and_laplace

    def forward(
        self,
        path: torch.Tensor,
        forces: torch.Tensor = None,
    ):
        """
        Args: path of shape [P, N, 3], forces of shape [P, N, 3]
        """
        
        f_n, laplace = self.force_and_laplace(
            path[:-1],
        )

        first_term = torch.square((path[1:] - path[:-1])) * (
            self.gamma / 4 / self.dt
        )
        second_term = torch.sum(torch.square(f_n) * (self.dt / 4 / self.gamma), axis=-1)

        third_term = laplace * self.dt * self.D / torch.tensor(2.0)

        # Result is the action and is expected to have shape [batch, 1]
        return first_term.sum(), second_term.sum(), third_term.sum()
