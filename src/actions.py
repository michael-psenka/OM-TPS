import torch

class SimpleAction(torch.nn.Module):
    def __init__(self, dt_xi):
        super(SimpleAction, self).__init__()
        self.dt_xi = dt_xi

    def forward(self, path: torch.Tensor, forces: torch.Tensor):
        """
        Simple Onsager-Machlup action. For details, see Eqn 7 from the paper:
        Artur B. Adib. Stochastic actions for diffusive dynamics: Reweight-
        ing, sampling, and minimization. The Journal of Physical Chemistry B,
        112(19):5910–5916, 05 2008

        Args:
            path: torch.Tensor of images of shape [N, C, H, W], where N is the number of images on the path.
            forces: torch.Tensor of diffusion model score estimates of shape [N, C, H, W], where N is the number of points on the path.
                
        Returns the OM action of the path (torch.Tensor of shape [1]).
        """
        result = 0.
        for i in range(path.shape[0] - 1):
            # (x_{i+1} - x_i)^2 / dt
            first_term = torch.square((path[i + 1, :] - path[i, :])) * (1 / self.dt_xi)

            f_n = forces[i]
            f_np = forces[i + 1]
            # (f_n^2 + f_{n+1}^2) * dt / 2 (squared force term)
            second_term = (torch.square(f_n) + torch.square(f_np)) * (self.dt_xi / 2.0)
            # (x_{i+1} - x_i) * (f_{n+1} - f_n) (force derivative term) 
            # # TODO: don't exactly understand this term (shouldn't we divide by the path difference)
            third_term = (path[i + 1, :] - path[i, :]) * (f_np - f_n)
            result = result + torch.sum(first_term + second_term + third_term)

        return result / torch.tensor(4.0)