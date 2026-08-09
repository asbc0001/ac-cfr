"""Readable full-tree CFR+ for Kuhn and Leduc poker."""

from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.solvers.naive_cfr import NaiveCFR, _validate_non_negative_integer


class NaiveCFRPlus(NaiveCFR):
    """CFR+ with aggregated regret clipping and delayed linear averaging."""

    def __init__(self, tree: IndexedGameTree, *, averaging_delay: int) -> None:
        _validate_non_negative_integer("averaging_delay", averaging_delay)
        super().__init__(tree)
        self._averaging_delay = averaging_delay

    @property
    def averaging_delay(self) -> int:
        """Return the number of initial iterations excluded from averaging."""
        return self._averaging_delay

    def _apply_regret_delta(self, regret_delta: list[list[float]]) -> None:
        # Clip only after every member history has contributed to the pass delta.
        for information_set_id, delta in enumerate(regret_delta):
            for action_position, value in enumerate(delta):
                cumulative_regret = self._regret_sum[information_set_id][action_position]
                self._regret_sum[information_set_id][action_position] = max(
                    0.0, cumulative_regret + value
                )

    def _averaging_weight(self, iteration: int) -> float:
        return float(max(iteration - self._averaging_delay, 0))
