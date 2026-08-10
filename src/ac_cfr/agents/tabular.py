"""Playable agent backed by a validated tabular average strategy."""

from ac_cfr.agents.base import PlayableAgent, Strategy
from ac_cfr.games.base import Action, InformationState
from ac_cfr.games.tabular import TabularGame
from ac_cfr.persistence.snapshots import LoadedTabularSnapshot


class TabularAgent(PlayableAgent):
    """Query one frozen Kuhn or Leduc average-policy table by information state."""

    __slots__ = ("_information_set_ids", "_snapshot", "_tabular_game")

    def __init__(
        self,
        tabular_game: TabularGame,
        snapshot: LoadedTabularSnapshot,
    ) -> None:
        if snapshot.metadata.game != tabular_game.game_id.value:
            raise ValueError("strategy snapshot and game do not match")
        self._tabular_game = tabular_game
        self._snapshot = snapshot
        self._information_set_ids = self._build_information_set_index()

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return the saved probabilities for one exact information state."""
        if not isinstance(information_state, InformationState):
            raise TypeError("information_state must be an InformationState")
        if information_state.game_id is not self._tabular_game.game_id:
            raise ValueError("information state belongs to a different game")
        if legal_actions != information_state.legal_actions:
            raise ValueError("legal_actions must match the information state")

        key = (information_state.player, information_state.encoding)
        try:
            information_set_id = self._information_set_ids[key]
        except KeyError as error:
            raise ValueError("information state is not present in the strategy snapshot") from error

        tree = self._tabular_game.tree
        offset = int(tree.information_set_action_offsets[information_set_id])
        count = int(tree.information_set_action_counts[information_set_id])
        saved_actions = tuple(
            Action(value) for value in tree.information_set_actions[offset : offset + count]
        )
        if legal_actions != saved_actions:
            raise ValueError("legal actions are incompatible with the strategy snapshot")
        return tuple(
            float(value) for value in self._snapshot.average_policy[offset : offset + count]
        )

    def _build_information_set_index(self) -> dict[tuple[int, tuple[int, ...]], int]:
        """Map each player-visible encoding to its stable information-set ID."""
        tree = self._tabular_game.tree
        index: dict[tuple[int, tuple[int, ...]], int] = {}
        for information_set_id, (player, offset, count) in enumerate(
            zip(
                tree.information_set_players,
                tree.information_set_encoding_offsets,
                tree.information_set_encoding_counts,
                strict=True,
            )
        ):
            start = int(offset)
            encoding = tuple(
                int(value) for value in tree.information_set_encodings[start : start + int(count)]
            )
            index[(int(player), encoding)] = information_set_id
        return index
