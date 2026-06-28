from collections import OrderedDict
from collections.abc import Sized
from itertools import pairwise
import random
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo, list_available_datasets

from helpers import (
    StandardScaler,
    cox_partial_loglik,
    concordance_index
)


def cox_loss(t: torch.Tensor,
              delta: torch.Tensor,
              loghr_hat: torch.Tensor):
    """Negative partial lok-likelihood + KL(q(theta) || p(theta))"""
    assert t.shape == delta.shape == loghr_hat.shape, "Tensor sizes must match."
    batch_size = len(t)
    # kl_weight = batch_size / N
    nll = -cox_partial_loglik(t, delta, loghr_hat)
    return nll


def train(model: nn.Module, loader: DataLoader, 
          optimizer: optim.Optimizer, n_epochs: int = 100) -> np.ndarray:
    losses = -10 * np.ones(n_epochs, dtype=np.float32) 
        # negative initialization to avoid 
        # ambiguities in case of early stopping
    assert isinstance(loader.dataset, Sized) # removes waves from PyLance
    N: int = len(loader.dataset)
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        # total_kl = 0
        for i, (x, t, delta) in enumerate(loader):
            # B = len(x)
            optimizer.zero_grad()
            loghr_hat = model(x)
            # model_kl = model.network_kl() 
            # scaled_kl = B/N * model_kl
            # loss = cox_elbo_loss(t, delta, loghr_hat.view(-1), scaled_kl)
            loss = cox_loss(t, delta, loghr_hat.view(-1))
            loss.backward()
            optimizer.step()
            # total_kl += scaled_kl.item()
            total_loss += loss.item()
            print(f"Epoch {epoch: 5d}, Batch {i: 3d}, Loss {loss.item():.3f}", end='\r') 
                  # f" (of which {scaled_kl.item():.3f} KL)", end='\r')
        avg_loss = total_loss / len(loader)
        # avg_kl = total_kl / len(loader)
        losses[epoch] = avg_loss
        print(f"Epoch {epoch: 5d}, Average loss {avg_loss:.3f}", 
              end = '\n' if epoch % 250 == 0 else '\r') 
              # (of which {avg_kl:.3f} KL)", end= '\n' if epoch % 250 == 0 else '\r')
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
    print(f"Total number of uncensored data: {aids.cid.sum()} ({100*aids.cid.mean():.2f}%)")

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

    train_data, val_data, test_data = random_split(dataset, 
                                                   (0.7, 0.1, 0.2), 
                                                   generator)

    loader = DataLoader(train_data, batch_size=128, generator=generator)
    # bnn = BayesianMLP(24, 72, 72, 1, prior_sigma=3.0) # here the topology is probably to change
    mlp_model = nn.Sequential(
        nn.Linear(24, 16),
        nn.Tanh(),
        # nn.Linear(72, 72),
        # nn.Tanh(),
        nn.Linear(16, 1)
    )
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    losses = train(mlp_model, loader, optimizer, n_epochs=1000)

    mlp_model.eval()
    test_time_np = time[test_data.indices].detach().numpy()
    test_delta_np = delta[test_data.indices].detach().numpy()
    loghaz_pred = mlp_model(design_matrix[test_data.indices]).view(-1).detach().numpy()
    cindex = concordance_index(test_time_np, -loghaz_pred, test_delta_np)
    print(f"\nC-index for a classic MLP with 24-16-1 architecture is {cindex:.4f}.")

    plt.plot(losses)
    plt.title("Training Loss (Only-Likelihood Approach)")
    plt.show()


if __name__ == '__main__':
    main_script()