"""Dense full-tree CFR+ for Kuhn and Leduc poker."""

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.solvers.cfr import CFR, _validate_non_negative_integer


class CFRPlus(CFR):
    """CFR+ with aggregated clipping and delayed linear averaging."""

    _CLIP_REGRETS = True

    def __init__(self, tree: IndexedGameTree, *, averaging_delay: int) -> None:
        _validate_non_negative_integer("averaging_delay", averaging_delay)
        super().__init__(tree)
        self._averaging_delay = averaging_delay

    @property
    def averaging_delay(self) -> int:
        """Return the number of initial iterations excluded from averaging."""
        return self._averaging_delay

    def _validate_restored_regrets(self, regrets: NDArray[np.float64]) -> None:
        if np.any(regrets < 0.0):
            raise ValueError("CFR+ regret_sum must not contain negative values")

    def _averaging_weight(self, iteration: int) -> float:
        return float(max(iteration - self._averaging_delay, 0))
