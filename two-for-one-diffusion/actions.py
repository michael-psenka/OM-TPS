import torch


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
        self.gamma = gamma.unsqueeze(0).unsqueeze(-1)  # shape of [1, n_atoms, 1]

    def forward(
        self,
        path: torch.Tensor,
        forces: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args: path of shape [B, P, N, 3], forces of shape [B, P, N, 3]
        """
        path_term = torch.square((path[:, 1:] - path[:, :-1])) / (2 * self.dt)
        force_term = torch.square(forces) * (self.dt / (2 * self.gamma**2))

        # mask out padded indices
        if mask is not None:
            mask = mask.unsqueeze(0).repeat(path_term.shape[0], 1)
        else:
            mask = torch.ones_like(path_term).bool()

        return path_term[mask].sum(), force_term[mask].sum(), torch.tensor(0).to(torch.float32)


class S2Action(torch.nn.Module):
    #TODO: remove this class since we always use HutchinsonAction anyways
    """Action with Hessian"""

    def __init__(self, force_func, dt, gamma, laplace_func, D=None):
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
        self.gamma = gamma.unsqueeze(0).unsqueeze(-1)  # shape of [1, n_atoms, 1]
        self.D = D

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [B, P, N, 3]
        """

        first_term = torch.square((path[:, 1:] - path[:, :-1])) / (2 * self.dt)
        second_term = torch.square(self.force_func(path[:, :-1])) * (
            self.dt / (2 * self.gamma**2)
        )
        third_term = self.laplace_func(path[:, :-1]) * self.dt * self.D / self.gamma
        # return result
        return (
            first_term.sum(),
            second_term.sum(),
            third_term.sum(),
        )  # @Sanjeev: how come we summed 2&3 into second slot before?


class HutchinsonAction(torch.nn.Module):
    """
    Action that is the same as S2 Action
    but calculates laplacian using Hutchinson trick using
    random vector of -1s and 1s.
    """

    def __init__(self, force_func, dt, gamma, D, N=1):
        super(HutchinsonAction, self).__init__()
        self.dt = dt
        self.gamma = gamma.unsqueeze(0).unsqueeze(-1)  # shape of [1, n_atoms, 1]
        self.D = D
        self.N = N
        self.force_func = force_func

        def force_and_laplace(
            x: torch.tensor, forces: torch.tensor = None, subsample_dimensions=None
        ):
            result = 0
            if forces is None:
                path_batch_flattened = x.reshape(-1, x.shape[-2], x.shape[-1])
                batch_forces = self.force_func(path_batch_flattened)
                # then reshape back to the original shape
                forces = batch_forces.reshape(
                    x.shape[0],
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                )

            if subsample_dimensions is not None:
                forces = forces.reshape(forces.shape[0], -1).gather(
                    -1, subsample_dimensions.expand(forces.shape[0], -1)
                )

            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                v = torch.randint(0, 2, forces.shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                # Calculate matrix vector product of Hessian and random vector
                (Av,) = torch.autograd.grad(
                    forces, x, grad_outputs=v, retain_graph=True, create_graph=True
                )
                if subsample_dimensions is not None:
                    Av = Av.reshape(Av.shape[0], -1).gather(
                        -1, subsample_dimensions.expand(forces.shape[0], -1)
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
        Args: path of shape [B, P, N, 3], forces of shape [B, P, N, 3]
        """
        num_atoms = path.shape[-2]
        
        f_n, laplace = self.force_and_laplace(
            path[:, :-1],
            forces,
            subsample_dimensions,
        )


        first_term = torch.square((path[:, 1:] - path[:, :-1])) / (2 * self.dt)

        second_term = torch.square(f_n) * (self.dt / (2 * self.gamma**2))

        third_term = laplace * self.dt * self.D / self.gamma

        # mask out padded indices
        if mask is not None:
            mask = mask.unsqueeze(0).repeat(first_term.shape[0], 1)
        else:
            mask = torch.ones_like(first_term).bool()


        # Result is the action and is expected to have shape [batch, 1]
        return first_term[mask].sum(), second_term[mask].sum(), third_term[mask].sum()


