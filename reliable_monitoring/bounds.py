"""
Based on:
https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py
and
https://github.com/bracha-laufer/pareto-testing
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def binomial(r_hat: np.ndarray | float, n: int, p: float, tail: bool = False) -> np.ndarray:
    """Compute binomial probabilities.

    Handles both scalar and array inputs, always returns array.

    Args:
        r_hat: Empirical risk(s), in [0, 1]. Can be scalar or array.
        n: Number of trials (sample size).
        p: Probability parameter.
        tail: If True, returns P(X >= k). If False, returns P(X <= k).

    Returns:
        Array of probabilities. If input is scalar, returns array of length 1.
    """
    r_hat_array = np.atleast_1d(r_hat)
    k_values = np.ceil(n * r_hat_array).astype(int)

    if tail:
        p_values = np.array([binom.sf(k - 1, n, p) for k in k_values])
    else:
        p_values = np.array([binom.cdf(k, n, p) for k in k_values])

    return p_values


def h1(y, mu):
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


def h2(y):
    return (1 + y) * np.log(1 + y) - y


### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu, x), mu)


def hoeffding_minus(mu, x, n):
    return -n * h1(np.minimum(mu, x), mu)


def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1


def bentkus_minus(mu, x, n):
    return np.log(max(binom.cdf(np.ceil(n * x), n, mu), 1e-10)) + 1


def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    def h1(y, mu):
        with np.errstate(all="ignore"):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    hoeffding_p_value = np.exp(-n * h1(np.minimum(r_hat, alpha), alpha))
    return np.fmin(bentkus_p_value, hoeffding_p_value)


def HB_mu_plus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)


def HB_mu_minus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_minus(mu, muhat, n)
        bentkus_mu = bentkus_minus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1e-10) > 0:
        return 0
    else:
        return brentq(_tailprob, 1e-10, muhat, maxiter=maxiters)
