"""Simple frozen policies used as untrained comparison agents."""

from ac_cfr.agents.base import PlayableAgent, Strategy
from ac_cfr.games.base import Action, InformationState


class BaselineAgent(PlayableAgent):
    """Uniform-random policy over the currently legal actions."""

    __slots__ = ()

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return a uniform distribution using no hidden game information."""
        if not isinstance(information_state, InformationState):
            raise TypeError("information_state must be an InformationState")
        if legal_actions != information_state.legal_actions:
            raise ValueError("legal_actions must match the information state")
        probability = 1.0 / len(legal_actions)
        return tuple(probability for _ in legal_actions)
