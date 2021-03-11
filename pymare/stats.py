"""Miscellaneous statistical functions."""

import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize, Bounds

from . import utils


def weighted_least_squares(y, v, X, tau2=0., return_cov=False):
    """2-D weighted least squares.

    Args:
        y (NDArray): 2-d array of estimates (studies x parallel datasets)
        v (NDArray): 2-d array of sampling variances
        X (NDArray): Fixed effect design matrix
        tau2 (float): tau^2 estimate to use for weights
        return_cov (bool): Whether or not to return the inverse cov matrix

    Returns:
        If return_cov is True, returns both fixed parameter estimates and the
        inverse covariance matrix; if False, only the parameter estimates.
    """

    w = 1. / (v + tau2)

    # Einsum indices: k = studies, p = predictors, i = parallel iterates
    wX = np.einsum('kp,ki->ipk', X, w)
    cov = wX.dot(X)

    # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims, so we
    # can vectorize computation along the second dimension (parallel datasets)
    precision = np.linalg.pinv(cov).T

    pwX = np.einsum('ipk,qpi->iqk', wX, precision)
    beta = np.einsum('ipk,ik->ip', pwX, y.T).T

    return (beta, precision) if return_cov else beta


def ensure_2d(arr):
    """Ensure the passed array has 2 dimensions."""
    if arr is None:
        return arr
    try:
        arr = np.array(arr)
    except:
        return arr
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def q_profile(y, v, X, alpha=0.05):
    """Get the CI for tau^2 via the Q-Profile method (Viechtbauer, 2007).

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        alpha (float, optional): alpha value defining the coverage of the CIs,
            where width(CI) = 1 - alpha. Defaults to 0.05.

    Returns:
        A dictionary with keys 'ci_l' and 'ci_u', corresponding to the lower
        and upper bounds of the tau^2 confidence interval, respectively.

    Notes:
        Following the Viechtbauer implementation, this method returns the
        interval that gives an equal probability mass at both tails (i.e.,
        P(tau^2 <= lower_bound)  == P(tau^2 >= upper_bound) == alpha/2), and
        *not* the smallest possible range of tau^2 values that provides the
        desired coverage.

    References:
        Viechtbauer, W. (2007). Confidence intervals for the amount of
        heterogeneity in meta-analysis. Statistics in Medicine, 26(1), 37-52.
    """
    k, p = X.shape
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (ensure_2d(y), ensure_2d(v), X)
    bds = Bounds([0], [np.inf], keep_feasible=True)

    # Use the D-L estimate of tau^2 as a starting point; when using a fixed
    # value, minimize() sometimes fails to stay in bounds.
    from .estimators import DerSimonianLaird
    ub_start = 2 * DerSimonianLaird().fit(y, v, X).params_['tau2']

    lb = minimize(lambda x: (q_gen(*args, x) - l_crit)**2, [0],
                  bounds=bds).x[0]
    ub = minimize(lambda x: (q_gen(*args, x) - u_crit)**2, [ub_start],
                  bounds=bds).x[0]
    return {'ci_l': lb, 'ci_u': ub}


def q_gen(y, v, X, tau2):
    """Generalized form of Cochran's Q-statistic.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        tau2 (float): Between-study variance. Must be >= 0.

    Returns:
        A float giving the value of Cochran's Q-statistic.

    References:
    Veroniki, A. A., Jackson, D., Viechtbauer, W., Bender, R., Bowden, J.,
    Knapp, G., Kuss, O., Higgins, J. P., Langan, D., & Salanti, G. (2016).
    Methods to estimate the between-study variance and its uncertainty in
    meta-analysis. Research synthesis methods, 7(1), 55–79.
    https://doi.org/10.1002/jrsm.1164
    """
    if np.any(tau2 < 0):
        raise ValueError("Value of tau^2 must be >= 0.")
    beta = weighted_least_squares(y, v, X, tau2)
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum(0)


def null_to_p(test_value, null_array, tail="two", symmetric=False):
    """Return p-value for test value(s) against null array.

    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    symmetric : bool
        When tail="two", indicates how to compute p-values. When False (default),
        both one-tailed p-values are computed, and the two-tailed p is double
        the minimum one-tailed p. When True, it is assumed that the null
        distribution is zero-centered and symmetric, and the two-tailed p-value
        is computed as P(abs(test_value) >= abs(null_array)).

    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).

    Notes
    -----
    P-values are clipped based on the number of elements in the null array.
    Therefore no p-values of 0 or 1 should be produced.

    When the null distribution is known to be symmetric and centered on zero,
    and two-tailed p-values are desired, use symmetric=True, as it is
    approximately twice as efficient computationally, and has lower variance.
    """
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    # For efficiency's sake, if there are more than 1000 values, pass only the unique
    # values through percentileofscore(), and then reconstruct.
    if len(test_value) > 1000:
        reconstruct = True
        test_value, uniq_idx = np.unique(test_value, return_inverse=True)
    else:
        reconstruct = False

    def compute_p(t, null):
        null = np.sort(null)
        idx = np.searchsorted(null, t, side="left").astype(float)
        return 1 - idx / len(null)

    if tail == "two":
        if symmetric:
            p = compute_p(np.abs(test_value), np.abs(null_array))
        else:
            p_l = compute_p(test_value, null_array)
            p_r = compute_p(test_value * -1, null_array * -1)
            p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(test_value * -1, null_array * -1)
    else:
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value)
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array))
    result = np.maximum(smallest_value, np.minimum(p, 1.0 - smallest_value))

    if reconstruct:
        result = result[uniq_idx]

    return result[0] if return_first else result


def nullhist_to_p(test_values, histogram_weights, histogram_bins):
    """Return one-sided p-value for test value against null histogram.

    Parameters
    ----------
    test_values : float or 1D array_like
        Values for which to determine p-value. Can be a single value or a one-dimensional array.
        If a one-dimensional array, it should have the same length as the histogram_weights' last
        dimension.
    histogram_weights : (B [x V]) array
        Histogram weights representing the null distribution against which test_value is compared.
        These should be raw weights or counts, not a cumulatively-summed null distribution.
    histogram_bins : (B) array
        Histogram bin centers. Note that this differs from numpy.histogram's behavior, which uses
        bin *edges*. Histogram bins created with numpy will need to be adjusted accordingly.

    Returns
    -------
    p_value : :obj:`float`
        P-value associated with the test value when compared against the null distribution.
        P-values reflect the probability of a test value at or above the observed value if the
        test value was drawn from the null distribution.
        This is a one-sided p-value.

    Notes
    -----
    P-values are clipped based on the largest observed non-zero weight in the null histogram.
    Therefore no p-values of 0 should be produced.
    """
    test_values = np.asarray(test_values)
    return_value = False
    if test_values.ndim == 0:
        return_value = True
        test_values = np.atleast_1d(test_values)
    assert test_values.ndim == 1
    assert histogram_bins.ndim == 1
    assert histogram_weights.shape[0] == histogram_bins.shape[0]
    assert histogram_weights.ndim in (1, 2)
    if histogram_weights.ndim == 2:
        assert histogram_weights.shape[1] == test_values.shape[0]
        voxelwise_null = True
    else:
        histogram_weights = histogram_weights[:, None]
        voxelwise_null = False

    n_bins = len(histogram_bins)
    inv_step = 1 / (histogram_bins[1] - histogram_bins[0])  # assume equal spacing

    # Convert histograms to null distributions
    # The value in each bin represents the probability of finding a test value
    # (stored in histogram_bins) of that value or lower.
    null_distribution = histogram_weights / np.sum(histogram_weights, axis=0)
    null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
    null_distribution /= np.max(null_distribution, axis=0)
    null_distribution = np.squeeze(null_distribution)

    smallest_value = np.min(null_distribution[null_distribution != 0])

    p_values = np.ones(test_values.shape)
    idx = np.where(test_values > 0)[0]
    value_bins = utils.round2(test_values[idx] * inv_step)
    value_bins[value_bins >= n_bins] = n_bins - 1  # limit to within null distribution

    # Get p-values by getting the value_bins-th value in null_distribution
    if voxelwise_null:
        # Pair each test value with its associated null distribution
        for i_voxel, voxel_idx in enumerate(idx):
            p_values[voxel_idx] = null_distribution[value_bins[i_voxel], voxel_idx]
    else:
        p_values[idx] = null_distribution[value_bins]

    # ensure p_value in the following range:
    # smallest_value <= p_value <= 1.0
    p_values = np.maximum(smallest_value, np.minimum(p_values, 1.0))
    if return_value:
        p_values = p_values[0]
    return p_values
