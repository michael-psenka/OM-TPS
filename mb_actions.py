import torch
from torch.func import vmap, grad, vjp


class S2Action(torch.nn.Module):
    """Action with Hessian"""

    def __init__(self, sample_force_func=None, laplace_func=None, dt=1.0, gamma=1.0, D=1.0):
        """
        Args:
            force_func: Force function
            laplace_func: Laplace function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(S2Action, self).__init__()
        self.force_func = vmap(sample_force_func)
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
        #f_np = self.force_func(x_np)

        first_term = torch.square((x_np - x_n)) * (self.gamma / 4 / self.dt)
        second_term = torch.square(f_n) * (
            self.dt / 4 / self.gamma
        )


        exact_laplace = self.laplace_func(x_n)

        third_term = exact_laplace * (
            self.dt * self.D / torch.tensor(2.0)
        )
        #third_term = 0.0
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
    Action with some randomness in hessian calculation. Saves one function call. 
    """

    def __init__(self, dt, gamma, D, sample_force_func=None, laplace_func = None):
        super(HutchinsonAction, self).__init__()
        self.sample_force_func = sample_force_func
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.true_laplace = laplace_func
        self.force_func = vmap(sample_force_func)

        
        def force_and_laplace(x: torch.tensor):
            N = 1
            res = torch.zeros_like(x)

            forces, vjp_func = vjp(self.sample_force_func, x)

            for _ in range(N+1):
                v = torch.randint(0, 2, (2,), dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                Av, = vjp_func(v)
                res += torch.sum(v*Av) 
            
            return forces, res / N

        self.force_and_laplace = vmap(force_and_laplace, randomness="different")

    
    def forward(self, path: torch.Tensor):
        
        x_n = path[:-1]
        x_np = path[1:]
        f_n = self.force_func(x_n)
        f_np, laplace = self.force_and_laplace(x_np)

        first_term = torch.square((path[1:] - path[:-1])) * (self.gamma / self.dt)

        second_term = (torch.square(f_n) + torch.square(f_np)) * (
            self.dt / self.gamma / 2.0
        )

        third_term = laplace * self.dt * self.D / torch.tensor(2.0)


        #print(first_term, second_term, third_term)
        #sdasd

        result = torch.sum(first_term + second_term + third_term)
        return result / torch.tensor(4.0)

