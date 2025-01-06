"""Track priors."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from stream_mapper.core import prior
from stream_mapper.core.prior import *  # noqa: F403
from stream_mapper.core.prior._track import ControlRegions as CoreControlRegions
from stream_mapper.core.prior._weight import HardThreshold as CoreHardThreshold
from stream_mapper.core.typing import ArrayNamespace  # noqa: TC001

from stream_mapper.pytorch.typing import Array

__all__ = prior.__all__


@dataclass(frozen=True, repr=False)
class HardThreshold(CoreHardThreshold[Array]):  # type: ignore[no-redef]
    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = field(default="torch", kw_only=True)  # type: ignore[arg-type]


@dataclass(frozen=True, repr=False)
class ControlRegions(CoreControlRegions[Array]):  # type: ignore[no-redef]
    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = field(default="torch", kw_only=True)  # type: ignore[arg-type]
