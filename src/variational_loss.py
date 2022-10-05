import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import kl
from torch import nn
from .moments import Moments

import time

class VariationalLoss:

    def __init__(self, D, K, hidden_dim, beta, cuda_device):

        # CUDA
        if cuda_device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda:"+str(cuda_device) if torch.cuda.is_available() else "cpu")
        
        # Encoder
        self.encoder = Moments(D, hidden_dim, K)
        self.encoder.to(self.device)
        # Decocer
        self.decoder = Moments(K, hidden_dim, D)
        self.decoder.to(self.device)

        # Parameters
        self.beta = torch.tensor(beta).to(self.device)


    def kl_term(self, v=0.5):

        prior_probs = (torch.ones(self.encoder.out.shape)*v).to(self.device)

        q = Bernoulli(self.encoder.out)
        p = Bernoulli(prior_probs)

        kl_div = kl.kl_divergence(q, p)

        kl_div = torch.sum(kl_div, dim=1)

        return kl_div


    def sample_from_q_z(self): 

        q_i = torch.clamp(self.encoder.out, 0, 1-1e-5)
        ones = torch.ones(q_i.shape).to(self.device)
        
        # Sample from U(0,1)
        ro = torch.rand(q_i.shape).to(self.device)

        # Reparameterization
        b = (ro+torch.exp(-self.beta)*(q_i-ro))/(ones-q_i) - ones
        c = -(q_i*torch.exp(-self.beta))/(ones-q_i)
        self.q_z = (-1/self.beta)*torch.log((-b + torch.sqrt(torch.pow(b, 2) - 4*c))/2)
        

    def reconstruction_term(self, x):  

        bce_loss = nn.BCELoss(reduction='none')

        self.decoder.forward(self.q_z)
        loss =  torch.sum(bce_loss(self.decoder.out, x), dim=1) # sum en D

        return -loss


    def ELBO(self, x):

        N = x.shape[0]

        self.encoder.forward(x)

        self.sample_from_q_z()

        reconstruction = self.reconstruction_term(x)

        kl_div = self.kl_term()

        elbo = torch.sum((reconstruction-kl_div), dim=0)/N

        self.ELBO_loss = -elbo
        self.kl = torch.sum(kl_div, dim=0)/N
        self.reconstruction = torch.sum(reconstruction, dim=0)/N

