"""Factories for the complete Kuhn and Leduc indexed games."""

from dataclasses import dataclass

from ac_cfr.common.config import GameConfigurationId, StateEncodingId
from ac_cfr.games.base import GameId
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree


@dataclass(frozen=True, slots=True)
class TabularGame:
    """One canonical small poker game and its compatibility identifiers."""

    game_id: GameId
    configuration_id: GameConfigurationId
    state_encoding_id: StateEncodingId
    tree: IndexedGameTree


def create_tabular_game(game_id: GameId) -> TabularGame:
    """Build the canonical indexed tree for Kuhn or Leduc poker."""
    if not isinstance(game_id, GameId):
        raise TypeError("game_id must be a GameId")
    if game_id is GameId.KUHN:
        configuration = KuhnConfig()
        tree = compile_game_tree(KuhnGame(), configuration)
    elif game_id is GameId.LEDUC:
        configuration = LeducConfig()
        tree = compile_game_tree(LeducGame(), configuration)
    else:
        raise ValueError("tabular training supports only Kuhn and Leduc")

    return TabularGame(
        game_id=game_id,
        configuration_id=configuration.configuration_id,
        state_encoding_id=configuration.state_encoding_id,
        tree=tree,
    )
