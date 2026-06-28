from collections import OrderedDict
from itertools import pairwise
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianGaussianLinear(nn.Module):
    """
    Parameters: mu_w, rho_w (weights), mu_b, rho_b (biases).
    sigma = softplus(rho) ensures positivity without constraints.
    """
    def __init__(self, in_dim: int, out_dim: int, prior_sigma: float = 2.0):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.prior_sigma  = prior_sigma

        kaiming_he_sigma = 4/(in_dim+out_dim)

        # Variational parameters for weights: shape (out, in)
        self.mu_w = nn.Parameter(torch.empty(out_dim, in_dim).normal_(0, kaiming_he_sigma))
        self.rho_w = nn.Parameter(torch.empty(out_dim, in_dim).fill_(-2.5))

        # Variational parameters for biases: shape (out,)
        self.mu_b = nn.Parameter(torch.empty(out_dim).normal_(0, 0.1))
        self.rho_b = nn.Parameter(torch.empty(out_dim).fill_(-2.5))


    def _sigma(self, rho: nn.Parameter):
        """Softplus riparametrization."""
        sigma = F.softplus(rho) + 1e-8
        return sigma


    def _kl_gaussian(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL Divergence (closed formula) between prior and 
        variational posterior with mean field assumption.
        """
        sigma0 = self.prior_sigma
        kl = (torch.log(sigma0/sigma) + (sigma**2 + mu**2) / (2*sigma0**2) - 1/2).sum()
        return kl
    

    def layer_kl(self) -> torch.Tensor:
        """KL divergence between the prior and 
        the variational posterior within the layer.
        """
        # Softplus transformation
        sigma_w = self._sigma(self.rho_w) # (out, in)
        sigma_b = self._sigma(self.rho_b) # (out,)
        return(
            self._kl_gaussian(self.mu_w, sigma_w) 
            + self._kl_gaussian(self.mu_b, sigma_b)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has size (B, nodes)

        # Softplus transformation
        sigma_w = self._sigma(self.rho_w) # (out, in)
        sigma_b = self._sigma(self.rho_b) # (out,)
        # Reparametrization
        eps_w = torch.randn_like(self.mu_w)
        eps_b = torch.randn_like(self.mu_b)
        W = sigma_w*eps_w + self.mu_w
        b  = sigma_b*eps_b + self.mu_b

        # Standard linear transformation
        out = x @ W.T + b

        # KL contribution from this layer
        # kl = self._kl_gaussian(self.mu_w, sigma_w) + self._kl_gaussian(self.mu_b, sigma_b)

        return out
    

class BayesianMLP(nn.Module):
    def __init__(self, *size: int, prior_sigma: float = 1.0):
        """Implementation of a Bayesian Neural Network
        trained with normal mean field variational assumption (by now).

        Attributes
        ----------
        size : Sequence[int]
            A sequence of dimensions of the layers. 
            First dimension must match the input, last dimension must match the output
        prior_sigma : float
            The standard deviation of the prior distribution of the nodes. Mean is zero.
        """
        if len(size) < 1:
            raise ValueError('At least one size (input size) must be provided')
        super().__init__()
        self.size = size
        self.prior_sigma = prior_sigma
        self.net = self.__make_net()


    def __make_net(self) -> nn.Sequential:
        layers: list[tuple[str, nn.Module]] = []
        for k, (s_in, s_out) in enumerate(pairwise(self.size[:-1])):
            layers.append((f"l{k}", BayesianGaussianLinear(s_in, s_out, self.prior_sigma)))
            layers.append((f"h{k}", nn.Tanh()))
        layers.append(('out', BayesianGaussianLinear(self.size[-2], self.size[-1], self.prior_sigma)))
        return nn.Sequential(OrderedDict(layers))

    def network_kl(self) -> torch.Tensor:
        """Returns the KL divergence between the prior and 
        the variational posterior as a 0-dim tensor, 
        by summing over all layer KL.
        """
        kl = torch.tensor(0.0, dtype=torch.float)
        for layer in self.net:
            if hasattr(layer, 'layer_kl') and isinstance(layer.layer_kl, Callable):
                kl += layer.layer_kl()
        return kl

    def forward(self, x):
        return self.net(x)