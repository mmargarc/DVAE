from .variational_loss import VariationalLoss
from torch import optim


class DVAE(VariationalLoss):

    def __init__(self, D, K, hidden_dim, beta=5, lr=1e-4, cuda_device=0):

        super().__init__(D, K, hidden_dim, beta, cuda_device)

        # Optimizer
        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=lr)


    def sgd_step(self, x):

        self.optim_encoder.zero_grad()
        self.optim_decoder.zero_grad()

        x = x.to(self.device)

        self.ELBO(x)

        self.ELBO_loss.backward()

        self.optim_encoder.step()
        self.optim_decoder.step()

