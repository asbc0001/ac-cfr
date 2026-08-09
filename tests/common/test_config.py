from ac_cfr.common.config import GameConfigurationId, StateEncodingId
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.kuhn import KuhnConfig
from ac_cfr.games.leduc import LeducConfig


def test_canonical_configuration_and_encoding_identifiers_are_stable() -> None:
    assert KuhnConfig().configuration_id is GameConfigurationId.KUHN
    assert KuhnConfig().state_encoding_id is StateEncodingId.KUHN
    assert LeducConfig().configuration_id is GameConfigurationId.LEDUC
    assert LeducConfig().state_encoding_id is StateEncodingId.LEDUC
    assert HoldemConfig.full().configuration_id is GameConfigurationId.HULHE
    assert HoldemConfig.modified().configuration_id is GameConfigurationId.MODIFIED_HULHE
    assert HoldemConfig.modified().state_encoding_id is StateEncodingId.HOLD_EM

    assert HoldemConfig.full(max_bets_per_round=2).configuration_id is None
    assert HoldemConfig(button_player=1).configuration_id is None
