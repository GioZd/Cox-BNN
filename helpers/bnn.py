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
    
class BayesianGaussianMultivariate(nn.Module):
    """
    Variant with full covariance for weights (suitable for small layers / last layer).
    Prior: N(0, sigma0^2 * I).
    Posterior: N(mu_w, Sigma_w) with Sigma_w = L @ L.T (Cholesky), plus mean-field bias.

    KL(N(mu, Sigma) || N(0, sigma0^2 I)) = 1/2 * [log(sigma0^{2n} / |Sigma|) + Tr(Sigma) / sigma0^2 + mu^T mu / sigma0^2 - n]
    """
    def __init__(self, in_dim: int, out_dim: int, prior_sigma: float = 2.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.prior_sigma = prior_sigma

        kaiming_he_sigma = 4/(in_dim+out_dim)
        n = out_dim * in_dim
                
        # Variational parameters for weights: shape (out, in)
        self.mu_w = nn.Parameter(torch.empty(out_dim, in_dim).normal_(0, kaiming_he_sigma))
        # Lower-triangular Cholesky factor of Sigma_w: shape (n, n)
        # Initialized as a small diagonal to stay close to the prior
        L_init = torch.eye(n) * 0.1
        self.L_w = nn.Parameter(L_init)

        # Variational parameters for biases: shape (out,)
        self.mu_b = nn.Parameter(torch.empty(out_dim).normal_(0, 0.1))
        self.rho_b = nn.Parameter(torch.empty(out_dim).fill_(-2.5))

    def _sigma(self, rho: nn.Parameter):
        """Softplus riparametrization."""
        sigma = F.softplus(rho) + 1e-8
        return sigma

    def _kl_gaussian(self, mu, sigma):
        """KL Divergence (closed formula) between prior and 
        variational posterior with mean field assumption.
        """
        sigma0 = self.prior_sigma
        kl = (torch.log(sigma0/sigma) + (sigma**2 + mu**2) / (2*sigma0**2) - 1/2).sum()
        return kl

    def _kl_gaussian_full(self) -> torch.Tensor:
        """
        Full-covariance KL divergence (closed form) between N(mu_w, Sigma_w) and the prior N(0, sigma0^2 I).
        Sigma_w = L @ L.T  (Cholesky decomposition guarantees SPD)

        KL(N(mu, Sigma) || N(0, sigma0^2 I)) = 1/2 * [log(sigma0^{2n} / |Sigma|) + Tr(Sigma) / sigma0^2 + mu^T mu / sigma0^2 - n]
        
        log|Sigma_w| = log|L L^T| = 2 * sum(log|diag(L)|)
        Tr(Sigma_w)  = ||L||_F^2  (since Tr(LL^T) = sum L_ij^2)
        """
        n = self.mu_w.numel()                                   # total number of weights
        sigma0 = self.prior_sigma                               # prior standard deviation
        L = self.L_w.tril()                                     # enforce lower-triangular structure on L
        diag_L = L.diagonal().abs().clamp(min=1e-8)             # absolute value of L's diagonal, clamped for numerical stability
        log_det_Sigma = 2.0 * diag_L.log().sum()                # log|Sigma| = log|LL^T| = 2 * sum(log|diag(L)|)
        trace_term = (L ** 2).sum() / (sigma0 ** 2)             # Tr(Sigma) / sigma0^2 = ||L||_F^2 / sigma0^2
        quad_term = (self.mu_w ** 2).sum() / (sigma0 ** 2)      # mu^T mu / sigma0^2
        
        # KL = 1/2 * [n*log(sigma0^2) - log|Sigma| + Tr(Sigma)/sigma0^2 + mu^T mu/sigma0^2 - n]
        kl = 0.5 * (n * 2 * torch.log(torch.tensor(sigma0)) - log_det_Sigma + trace_term + quad_term - n)
        return kl

    def layer_kl(self) -> torch.Tensor:
        """KL divergence between the prior and 
        the variational posterior within the layer.
        """
        sigma_b = self._sigma(self.rho_b)
        return(
            self._kl_gaussian_full() 
            + self._kl_gaussian(self.mu_b, sigma_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has size (B, in_dim)

        n = self.mu_w.numel()
        L = self.L_w.tril()

        # Reparametrization trick: w_flat = mu_flat + L @ eps,  eps ~ N(0, I)
        eps_w = torch.randn(n, device=x.device)
        w_flat = self.mu_w.view(-1) + L @ eps_w
        W = w_flat.view(self.out_dim, self.in_dim)

        # Softplus transformation + reparametrization for biases
        sigma_b = self._sigma(self.rho_b)
        eps_b = torch.randn_like(self.mu_b)
        b = self.mu_b + sigma_b * eps_b

        # Standard linear transformation
        out = x @ W.T + b

        return out

class BayesianMLP(nn.Module):
    def __init__(self, *size: int, prior_sigma: float = 1.0):
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
        layers.append(('out', BayesianGaussianMultivariate(self.size[-2], self.size[-1], self.prior_sigma)))
        return nn.Sequential(OrderedDict(layers))

    def network_kl(self) -> torch.Tensor:
        """Sum over all KL contributions in the layers."""
        kl = torch.tensor(0.0, dtype=torch.float)
        for layer in self.net:
            if hasattr(layer, 'layer_kl') and isinstance(layer.layer_kl, Callable):
                kl += layer.layer_kl()
        return kl

    def forward(self, x):
        return self.net(x)