"""Gaussian stream model."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch as xp

from stream_mapper.core.builtin import WhereRequiredError
from stream_mapper.core.builtin._stats.trunc_norm import logpdf as truncnorm_logpdf
from stream_mapper.core.builtin._stats.trunc_skewnorm import (
    logpdf as truncskewnorm_logpdf,
)

from stream_mapper.pytorch.builtin import SkewNormal

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params

    from stream_mapper.pytorch.typing import Array


@dataclass(unsafe_hash=True)
class TruncatedSkewNormal(SkewNormal):
    """Truncated Skew-Normal."""

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array

        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        if where is not None:
            idx = where[tuple(self.coord_bounds.keys())].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data), self.ndim), dtype=bool, require_grad=False)
            # This has shape (N,F) so will broadcast correctly.

        cns, cens = self.coord_names, self.coord_err_names
        x = data[cns].array  # (N, F)

        a_, b_ = self._get_lower_upper_bound(data[self.indep_coord_names].array)
        a, b = a_[idx], b_[idx]
        # a, b = _0 + self.xp.asarray([self.coord_bounds[k] for k in cns]).T[:, None, :]
        mu = self._stack_param(mpars, "mu", cns)[idx]
        ln_s = self._stack_param(mpars, "ln-sigma", cns)[idx]
        skew = self._stack_param(mpars, "skew", cns)[idx]
        if cens is not None:
            # it's fine if sigma_o is 0
            # TODO: I suspect there are better ways to write this
            sigma_o = data[cens].array[idx]
            sigma = self.xp.exp(ln_s)
            skew = skew * sigma / self.xp.sqrt(sigma**2 + (1 + skew**2) * sigma_o**2)
            ln_s = self.xp.log(sigma**2 + sigma_o**2) / 2

        # Find where -inf
        with xp.no_grad():
            _lpdf = truncskewnorm_logpdf(
                x[idx], loc=mu, ln_sigma=ln_s, skew=skew, a=a, b=b, xp=self.xp
            )
            fnt = xp.isfinite(_lpdf)  # apply to X[idx] only

        # Compute SN where it's finite, to avoid numerical issues with the gradient
        sn_lnpdf = self.xp.full_like(x[idx], 0)
        sn_lnpdf[fnt] = truncskewnorm_logpdf(
            x[idx][fnt],
            loc=mu[fnt],
            ln_sigma=ln_s[fnt],
            skew=skew[fnt],
            a=a[fnt],
            b=b[fnt],
            xp=self.xp,
        )

        # Compute normal where SN is infinite.
        # Subtract 100 b/c that's where the SN logpdf drops to -inf
        n_lnpdf = (
            truncnorm_logpdf(x[idx], loc=mu, ln_sigma=ln_s, a=a, b=b, xp=self.xp) - 100
        )

        idxlnliks = xp.where(fnt, sn_lnpdf, n_lnpdf)

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks[idx] = idxlnliks

        return lnliks.sum(1)
