import torch
from torch.func import vmap, grad, vjp


class S2Action(torch.nn.Module):
    """Action with Hessian calculated using autodiff - Hessian is exact."""

    def __init__(self, force_func=None, laplace_func=None, dt=1.0, gamma=1.0, D=1.0):
        """
        Args:
            force_func: Force function
            laplace_func: Laplace function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(S2Action, self).__init__()
        self.force_func = vmap(force_func)
        self.laplace_func = laplace_func
        self.dt = dt
        self.gamma = gamma
        self.D = D

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [P, 2]
        """

        x_n = path[:-1]
        x_np = path[1:]
        f_n = self.force_func(x_n)

        first_term = torch.sum(
            torch.square((x_np - x_n)) * (self.gamma / 4 / self.dt), axis=-1
        )
        second_term = torch.sum(torch.square(f_n) * (self.dt / 4 / self.gamma), axis=-1)

        exact_laplace = self.laplace_func(x_n)

        third_term = exact_laplace.squeeze() * (self.dt * self.D / torch.tensor(2.0))

        assert (
            first_term.shape == second_term.shape
            and second_term.shape == third_term.shape
        )
        result = torch.sum(first_term + second_term - third_term)

        # Result is the action and is expected to have shape [batch, 1]
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

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [P, 2]
        """
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / 4 / self.dt)
        second_term = torch.square(self.force_func(path[:-1])) * (
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

    def forward(self, path: torch.Tensor):
        # Args: path of shape [P, 2]
        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / self.dt)
        f_n = self.force_func(path[:-1])
        f_np = self.force_func(path[1:])
        second_term = (torch.square(f_n) + torch.square(f_np)) * (
            self.dt / self.gamma / 2.0
        )
        third_term = (path[1:] - path[:-1]) * (f_np - f_n)

        result = torch.sum(first_term + second_term + third_term)
        return result / torch.tensor(4.0)


class HutchinsonAction(torch.nn.Module):
    """
    Action that is the same as S2 Action but calculates laplacian using Hutchinson trick using
    random vector of -1s and 1s.
    """

    def __init__(
        self,
        dt,
        gamma,
        D,
        N=1,
        sample_force_func=None,
        laplace_func=None,
        force_func=None,
        diffusion_model=True,
    ):
        super(HutchinsonAction, self).__init__()

        sample_force_func = (
            force_func if sample_force_func is None else sample_force_func
        )

        self.sample_force_func = sample_force_func
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.N = N
        self.true_laplace = laplace_func
        self.force_func = vmap(sample_force_func)

        def force_and_laplace(x: torch.tensor):
            res = 0

            if diffusion_model:
                x = x.unsqueeze(0)

            forces, vjp_func = vjp(self.sample_force_func, x)

            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                shape = x.shape
                v = torch.randint(0, 2, shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                # Calculate matrix vector product of Hessian and random vector
                (Av,) = vjp_func(v)

                # Make it a scalar. Minus is from the sign of the forces
                res += -torch.sum(v * Av)

            return forces, res / N

        self.force_and_laplace = vmap(force_and_laplace, randomness="different")

    def forward(self, path: torch.Tensor):

        x_n = path[:-1]
        x_np = path[1:]
        f_n, laplace = self.force_and_laplace(x_n)

        # Make sure the terms are [batch, 1] as is the third term
        first_term = torch.sum(
            torch.square((x_np - x_n)) * (self.gamma / 4 / self.dt), axis=-1
        )
        second_term = torch.sum(torch.square(f_n) * (self.dt / 4 / self.gamma), axis=-1)

        third_term = laplace * self.dt * self.D / torch.tensor(2.0)

        # First apply physical dimension sum. The sum can be rearanged but Hutch term has to be scalar. So others need to be too
        result = torch.sum(first_term + second_term - third_term)

        # Result is the action and is expected to have shape [batch, 1]
        return result
