import torch


class CommittorNN(torch.nn.Module):
    def __init__(self, model):
        super(CommittorNN, self).__init__()
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h, t):
        # model returns per-residue contribution to the committor
        per_atom_contrib = self.model(x, h, t, return_energy=True)
        # sum over all atoms and pass through sigmoid to get committor probability
        committor_prob = self.sigmoid(per_atom_contrib.sum(dim=(-2, -1)))
        return committor_prob
