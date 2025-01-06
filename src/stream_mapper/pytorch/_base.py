"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch as xp
from torch import nn

from stream_mapper.core import ModelBase as CoreModelBase
from stream_mapper.core._connect.nn_namespace import NN_NAMESPACE
from stream_mapper.core._connect.xp_namespace import XP_NAMESPACE
from stream_mapper.core.typing import ArrayNamespace  # noqa: TC001
from stream_mapper.core.utils.dataclasses import ArrayNamespaceReprMixin
from stream_mapper.core.utils.scale import names_intersect

from stream_mapper.pytorch.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_mapper.core import Data

    Self = TypeVar("Self", bound="ModelBase")


@dataclass(unsafe_hash=True, repr=False)
class ModelBase(nn.Module, CoreModelBase[Array, NNModel]):
    """Model base class."""

    _: KW_ONLY
    array_namespace: ArrayNamespace[Array] = xp

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        self: Self = super().__new__(cls, *args, **kwargs)  # <- CoreModelBase

        # PyTorch needs to be initialized before attributes are assigned.
        nn.Module.__init__(self)
        return self

    def __post_init__(self) -> None:
        super().__post_init__()

        # Net needs to added to ensure that it's registered as a module.
        # TODO! not need to overwrite the descriptor.
        self.net: NNModel = self.net

    def __repr__(self) -> str:
        """Repr."""
        return ArrayNamespaceReprMixin.__repr__(self)

    # __setstate__ = SupportsXPNN[Array, NNModel].__setstate__
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state."""
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)
        object.__setattr__(self, "array_namespace", XP_NAMESPACE[self.array_namespace])
        object.__setattr__(self, "_nn_namespace_", NN_NAMESPACE[self.array_namespace])

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        (N, 3) Array
            fraction, mean, ln-sigma

        """
        if self.net is None:
            return self.xp.asarray([])

        # The forward step runs on the normalized coordinates
        scaled_data = self.data_scaler.transform(
            data, names=names_intersect(data, self.data_scaler), xp=self.xp
        )
        return self._forward_priors(
            self.net(scaled_data[self.indep_coord_names].array), scaled_data
        )
