"""Gaussian stream model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
import math
from typing import TYPE_CHECKING

from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.pytorch._base import ModelBase
from stream_ml.pytorch.builtin._normal import norm_logpdf
from stream_ml.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace

__all__: list[str] = []


_sqrt2 = math.sqrt(2)


def skewnorm_logpdf(
    value: Array, /, loc: Array, sigma: Array, skew: Array, *, xp: ArrayNamespace
) -> Array:
    r"""Log of the probability density function of the normal distribution.

    Parameters
    ----------
    value : Array
        Value at which to evaluate the PDF.
    loc : Array
        Mean of the distribution.
    sigma : Array
        Variance of the distribution.
    skew : Array
        Skewness of the distribution.

    xp : ArrayNamespace, keyword-only
        Array namespace.

    Returns
    -------
    Array
        Log of the PDF.
    """
    return norm_logpdf(value, loc=loc, sigma=sigma, xp=xp) + xp.log(
        1 + xp.erf(skew * (value - loc) / sigma / _sqrt2)
    )


@dataclass(unsafe_hash=True)
class SkewNormal(ModelBase):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(((..., ("mu", "sigma", "skew")),))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array]({})

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    def _net_init_default(self) -> NNModel:
        raise NotImplementedError

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        return skewnorm_logpdf(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.clip(mpars[c, "sigma"], 1e-10),
            skew=self.xp.clip(mpars[c, "skew"], 1e-10),
            xp=self.xp,
        )


# ============================================================================


def log_truncation_term(
    ab: tuple[float | Array, float | Array],
    /,
    loc: Array,
    sigma: Array,
    skew: Array,
    *,
    xp: ArrayNamespace,
) -> Array:
    r"""Integral of skew-normal from a to b.

    Parameters
    ----------
    ab : tuple[Array, Array]
        lower, upper at which to evaluate the CDF.
    loc : Array
        Mean of the distribution.
    sigma : Array
        Variance of the distribution.
    skew : Array
        Skewness of the distribution.

    xp : ArrayNamespace, keyword-only
        Array namespace.

    Returns
    -------
    Array
    """
    erfa = xp.erf((ab[0] - loc) / sigma / _sqrt2)
    erfb = xp.erf((ab[1] - loc) / sigma / _sqrt2)
    return xp.log(erfb - erfa) + xp.log(erfb + erfa + 2) - xp.log(4)


def truncatedskewnorm_logpdf(
    value: Array,
    /,
    loc: Array,
    sigma: Array,
    skew: Array,
    ab: tuple[float | Array, float | Array],
    *,
    xp: ArrayNamespace,
) -> Array:
    r"""Integral of skew-normal from a to b.

    Parameters
    ----------
    value : Array
        Value at which to evaluate the PDF.
    loc : Array
        Mean of the distribution.
    sigma : Array
        Variance of the distribution.
    skew : Array
        Skewness of the distribution.
    ab : tuple[Array, Array]
        lower, upper at which to evaluate the CDF.

    xp : ArrayNamespace, keyword-only
        Array namespace.

    Returns
    -------
    Array
    """
    return skewnorm_logpdf(
        value, loc=loc, sigma=sigma, skew=skew, xp=xp
    ) - log_truncation_term(ab, loc=loc, sigma=sigma, skew=skew, xp=xp)


@dataclass(unsafe_hash=True)
class TruncatedSkewNormal(ModelBase):
    r"""1D Gaussian with mixture weight.

    :math:`(weight, \mu, \sigma)(\phi1)`

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(((..., ("mu", "sigma", "skew")),))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array]({})

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    def _net_init_default(self) -> NNModel:
        raise NotImplementedError

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        return truncatedskewnorm_logpdf(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.clip(mpars[c, "sigma"], 1e-10),
            skew=self.xp.clip(mpars[c, "skew"], 1e-10),
            ab=self.coord_bounds[self.coord_names[0]],
            xp=self.xp,
        )
