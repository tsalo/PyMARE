"""Transforms."""
import numpy as np
from scipy import stats

from . import references
from .due import due


def sample_sizes_to_dof(sample_sizes):
    """Calculate degrees of freedom from a list of sample sizes using a simple heuristic.

    Parameters
    ----------
    sample_sizes : array_like
        A list of sample sizes for different groups in the study.

    Returns
    -------
    dof : int
        An estimate of degrees of freedom. Number of participants minus number
        of groups.
    """
    dof = np.sum(sample_sizes) - len(sample_sizes)
    return dof


def sample_sizes_to_sample_size(sample_sizes):
    """Calculate appropriate sample size from a list of sample sizes using a simple heuristic.

    Parameters
    ----------
    sample_sizes : array_like
        A list of sample sizes for different groups in the study.

    Returns
    -------
    sample_size : int
        Total (sum) sample size.
    """
    sample_size = np.sum(sample_sizes)
    return sample_size


def sd_to_varcope(sd, sample_size):
    """Convert standard deviation to sampling variance.

    Parameters
    ----------
    sd : array_like
        Standard deviation of the sample
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter
    """
    se = sd / np.sqrt(sample_size)
    varcope = se_to_varcope(se)
    return varcope


def se_to_varcope(se):
    """Convert standard error values to sampling variance.

    Parameters
    ----------
    se : array_like
        Standard error of the sample parameter

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter

    Notes
    -----
    Sampling variance is standard error squared.
    """
    varcope = se ** 2
    return varcope


def samplevar_dataset_to_varcope(samplevar_dataset, sample_size):
    """Convert "sample variance of the dataset" to "sampling variance".

    Parameters
    ----------
    samplevar_dataset : array_like
        Sample variance of the dataset (i.e., variance of the individual observations in a single
        sample). Can be calculated with ``np.var``.
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter (i.e., variance of sampling distribution for the
        parameter).

    Notes
    -----
    Sampling variance is sample variance divided by sample size.
    """
    varcope = samplevar_dataset / sample_size
    return varcope


def t_and_varcope_to_beta(t, varcope):
    """Convert t-statistic to parameter estimate using sampling variance.

    Parameters
    ----------
    t : array_like
        T-statistics of the parameter
    varcope : array_like
        Sampling variance of the parameter

    Returns
    -------
    beta : array_like
        Parameter estimates
    """
    beta = t * np.sqrt(varcope)
    return beta


def t_and_beta_to_varcope(t, beta):
    """Convert t-statistic to sampling variance using parameter estimate.

    Parameters
    ----------
    t : array_like
        T-statistics of the parameter
    beta : array_like
        Parameter estimates

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter
    """
    varcope = (beta / t) ** 2
    return varcope


def p_to_z(p, tail="two"):
    """Convert p-values to (unsigned) z-values.

    Parameters
    ----------
    p : array_like
        P-values
    tail : {'one', 'two'}, optional
        Whether p-values come from one-tailed or two-tailed test. Default is
        'two'.

    Returns
    -------
    z : array_like
        Z-statistics (unsigned)
    """
    p = np.array(p)
    if tail == "two":
        z = stats.norm.isf(p / 2)
    elif tail == "one":
        z = stats.norm.isf(p)
        z = np.array(z)
        z[z < 0] = 0
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    if z.shape == ():
        z = z[()]
    return z


@due.dcite(references.T2Z_TRANSFORM, description="Introduces T-to-Z transform.")
@due.dcite(
    references.T2Z_IMPLEMENTATION,
    description="Python implementation of T-to-Z transform.",
)
def t_to_z(t_values, dof):
    """Convert t-statistics to z-statistics.

    An implementation of [1]_ from Vanessa Sochat's TtoZ package [2]_.

    Parameters
    ----------
    t_values : array_like
        T-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    z_values : array_like
        Z-statistics

    References
    ----------
    .. [1] Hughett, P. (2007). Accurate Computation of the F-to-z and t-to-z
           Transforms for Large Arguments. Journal of Statistical Software,
           23(1), 1-5.
    .. [2] Sochat, V. (2015, October 21). TtoZ Original Release. Zenodo.
           http://doi.org/10.5281/zenodo.32508
    """
    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values_nonzero = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = nonzero <= c
    k2 = nonzero > c

    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = stats.t.cdf(t1, df=dof)
    z_values_t1 = stats.norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values_nonzero[k1] = z_values_t1
    z_values_nonzero[k2] = z_values_t2

    z_values = np.zeros(t_values.shape)
    z_values[t_values != 0] = z_values_nonzero
    return z_values


def z_to_t(z_values, dof):
    """Convert z-statistics to t-statistics.

    An inversion of the t_to_z implementation of [1]_ from Vanessa Sochat's
    TtoZ package [2]_.

    Parameters
    ----------
    z_values : array_like
        Z-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    t_values : array_like
        T-statistics

    References
    ----------
    .. [1] Hughett, P. (2007). Accurate Computation of the F-to-z and t-to-z
           Transforms for Large Arguments. Journal of Statistical Software,
           23(1), 1-5.
    .. [2] Sochat, V. (2015, October 21). TtoZ Original Release. Zenodo.
           http://doi.org/10.5281/zenodo.32508
    """
    # Select just the nonzero voxels
    nonzero = z_values[z_values != 0]

    # We will store our results here
    t_values_nonzero = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = nonzero <= c
    k2 = nonzero > c

    # Subset the data into two sets
    z1 = nonzero[k1]
    z2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_z1 = stats.norm.cdf(z1)
    t_values_z1 = stats.t.ppf(p_values_z1, df=dof)

    # Calculate p values for > 0
    p_values_z2 = stats.norm.cdf(-z2)
    t_values_z2 = -stats.t.ppf(p_values_z2, df=dof)
    t_values_nonzero[k1] = t_values_z1
    t_values_nonzero[k2] = t_values_z2

    t_values = np.zeros(z_values.shape)
    t_values[z_values != 0] = t_values_nonzero
    return t_values
