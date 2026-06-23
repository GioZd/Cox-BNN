import torch

from lifelines.utils import concordance_index

def get_riskset(t: torch.Tensor) -> torch.Tensor:
    """Given a 1-dim pytorch Tensor of event times of size N,
    returns a boolean matrix N x N where row i contains the list
    of values (boolean mask) that are greater or equal to t[i].
    """
    return t.view(-1, 1) <= t.repeat(len(t), 1)


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
    riskset = get_riskset(t) 
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
