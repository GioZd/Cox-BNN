from collections import OrderedDict
from collections.abc import Sized
from itertools import pairwise
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo, list_available_datasets

from helpers import (
    BayesianMLP,
    StandardScaler
)


def cox_partial_loglik(t: torch.Tensor, 
                       delta: torch.Tensor, 
                       loghr_hat: torch.Tensor) -> torch.Tensor:
    """Cox Partial Likelihood (the negative is a loss component)

    Parameters
    ----------
    t : Tensor[float]
        Time to event or censoring
    delta : Tensor[float]
        Indicator `1.0` is observed
    loghr_hat : Tensor[float]
        Output of the BNN
    
    All inputs must be reduced to size (BATCH_SIZE,).
    
    Returns
    -------
    ll : ScalarTensor
        The partial log-likelihood for a Cox proportional hazards model
        with Breslow tie-breaker.

        Sum_{i: delta_i==1} [h - log Sum_{j: t_j>=t_i} exp(h)]

        h output of the neural network.
    """
    assert len(t.shape) == len(delta.shape) == len(loghr_hat.shape) == 1, "All inputs must be reduced to size (BATCH_SIZE,)"
    hr_hat = torch.exp(loghr_hat)
    riskset = t.view(-1, 1) <= t.repeat(len(t), 1) 
        # boolean matrix where the `i`-th row denotes the
        # risk set of the `i`-th instance, i.e. the indices `j`
        # for which the observer time `y_j >= y_i`.

    # ll = loghr_hat.where(delta>0, 0.0).sum()
    # for t_i in t[delta>0]:
    #     ll -= torch.log(
    #         hr_hat.where(t>=t_i, 0.0).sum()
    #     )

    log_denominator = torch.log(
        (riskset*hr_hat.repeat(len(hr_hat), 1)).sum(dim=1)
    )
    ll = ((delta > 0) * (loghr_hat - log_denominator)).sum()

    return ll

def cox_elbo_loss(t: torch.Tensor,
              delta: torch.Tensor,
              loghr_hat: torch.Tensor,
              kl: torch.Tensor):
    """Negative partial lok-likelihood + KL(q(theta) || p(theta))"""
    assert t.shape == delta.shape == loghr_hat.shape, "Tensor sizes must match."
    batch_size = len(t)
    # kl_weight = batch_size / N
    nll = -cox_partial_loglik(t, delta, loghr_hat)
    return nll + kl

def train(model: BayesianMLP, loader: DataLoader, 
          optimizer: optim.Optimizer, n_epochs: int = 100) -> np.ndarray:
    losses = -10 * np.ones(n_epochs, dtype=np.float32) 
        # negative initialization to avoid 
        # ambiguities in case of early stopping
    assert isinstance(loader.dataset, Sized) # removes waves from PyLance
    N: int = len(loader.dataset)
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        total_kl = 0
        for i, (x, t, delta) in enumerate(loader):
            B = len(x)
            optimizer.zero_grad()
            loghr_hat = model(x)
            model_kl = model.network_kl() 
            scaled_kl = B/N * model_kl
            loss = cox_elbo_loss(t, delta, loghr_hat.view(-1), scaled_kl)
            loss.backward()
            optimizer.step()
            total_kl += scaled_kl.item()
            total_loss += loss.item()
            print(f"Epoch {epoch: 5d}, Batch {i: 3d}, Loss {loss.item():.3f}" 
                  f" (of which {scaled_kl.item():.3f} KL)", end='\r')
        avg_loss = total_loss / len(loader)
        avg_kl = total_kl / len(loader)
        losses[epoch] = avg_loss
        print(f"Epoch {epoch: 5d}, Average loss {avg_loss:.3f} (of which {avg_kl:.3f} KL)", end= '\n' if epoch % 250 == 0 else '\r')
    return losses


def main_script():
    aids_clinical_trials_group = fetch_ucirepo(id=890) 
    if aids_clinical_trials_group.data is None:
        raise TypeError('Could not retrieve the dataset. Dataset is NoneType.')
    aids: pd.DataFrame = pd.concat([
            aids_clinical_trials_group.data['features'], 
            aids_clinical_trials_group.data['targets']
        ], axis=1
    )

    print('Total number of observations:', len(aids))
    print(f"Total number of censored data: {aids.cid.sum()} ({100*aids.cid.mean():.2f}%)")

    scaler = StandardScaler()
    design_matrix = torch.tensor(
        pd.get_dummies(
            aids.drop(columns=['time', 'zprior', 'cid']),
            columns=['trt'],
            dtype=np.float64
        ).values,
        dtype = torch.float
    )

    design_matrix = scaler.fit_transform(design_matrix)
    time = torch.tensor(aids.time, dtype=torch.float)
    delta = torch.tensor(aids.cid)

    dataset = TensorDataset(design_matrix, time, delta)
    generator = torch.Generator()
    generator.manual_seed(42) # other seeds are possible
    loader = DataLoader(dataset, batch_size=128, generator=generator)
    bnn = BayesianMLP(24, 72, 72, 1, prior_sigma=3.0) # here the topology is probably to change
    optimizer = optim.Adam(bnn.parameters(), lr=0.001)
    losses = train(bnn, loader, optimizer, n_epochs=1000)
    plt.plot(losses)
    plt.title("Training ELBO Loss")
    plt.show()


if __name__ == '__main__':
    main_script()