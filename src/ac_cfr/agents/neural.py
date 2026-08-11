"""Playable agent backed by a validated Deep CFR strategy network."""

import torch

from ac_cfr.agents.base import PlayableAgent, Strategy, normalise_strategy
from ac_cfr.games.base import Action, GameId, InformationState
from ac_cfr.games.leduc_neural import encode_leduc_information_state
from ac_cfr.persistence.deep_cfr_snapshots import LoadedDeepCFRSnapshot


class NeuralAgent(PlayableAgent):
    """Query one frozen Leduc average-strategy network."""

    __slots__ = ("_snapshot",)

    def __init__(self, snapshot: LoadedDeepCFRSnapshot) -> None:
        if not isinstance(snapshot, LoadedDeepCFRSnapshot):
            raise TypeError("snapshot must be a LoadedDeepCFRSnapshot")
        self._snapshot = snapshot

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return masked network probabilities in legal-action order."""
        if not isinstance(information_state, InformationState):
            raise TypeError("information_state must be an InformationState")
        if information_state.game_id is not GameId.LEDUC:
            raise ValueError("information state must belong to Leduc")
        if legal_actions != information_state.legal_actions:
            raise ValueError("legal_actions must match the information state")

        network = self._snapshot.network
        device = next(network.parameters()).device
        state = torch.from_numpy(encode_leduc_information_state(information_state).copy())
        action_indices = torch.tensor(
            [int(action) for action in legal_actions],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            logits = network(state.to(device).unsqueeze(0))[0]
            legal_logits = logits.index_select(0, action_indices)
            if not bool(torch.isfinite(legal_logits).all()):
                raise FloatingPointError("strategy network logits must be finite")
            probabilities = torch.softmax(legal_logits, dim=0)
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError("strategy network probabilities must be finite")
        return normalise_strategy(
            tuple(float(value) for value in probabilities.cpu()), legal_actions
        )
