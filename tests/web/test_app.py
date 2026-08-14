import json
from pathlib import Path
from random import Random
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI

from ac_cfr.agents import BaselineAgent, RuleBasedAgent
from ac_cfr.common.config import (
    DeepCFRImplementationId,
    GameConfigurationId,
    ModelConfigId,
    StateEncodingId,
)
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.models import build_deep_cfr_network
from ac_cfr.persistence.deep_cfr_snapshots import export_deep_cfr_snapshot, load_deep_cfr_snapshot
from ac_cfr.persistence.registry import load_strategy_registry
from ac_cfr.persistence.snapshots import file_sha256
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.web.app import create_app
from ac_cfr.web.gameplay import HandNotFoundError, HandStore

_PROJECT_ROOT = Path(__file__).parents[2]
_REGISTRY_PATH = _PROJECT_ROOT / "configs" / "strategy_registry.json"


def _app(seed: int) -> FastAPI:
    return create_app(
        registry_path=_REGISTRY_PATH,
        project_root=_PROJECT_ROOT,
        master_rng=Random(seed),
    )


@pytest.fixture
def anyio_backend() -> str:
    """Use the application's asyncio backend for ASGI tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_registry_exposes_only_the_curated_matrix_and_resolves_holdem_baselines() -> None:
    registry = load_strategy_registry(_REGISTRY_PATH, project_root=_PROJECT_ROOT)

    async with _client(_app(1)) as client:
        response = await client.get("/api/strategies")

    assert response.status_code == 200
    strategies = {entry["strategy_id"]: entry for entry in response.json()}
    assert set(strategies) == {
        "kuhn_random",
        "kuhn_cfr_final",
        "kuhn_cfr_plus_final",
        "leduc_random",
        "leduc_cfr_final",
        "leduc_cfr_plus_final",
        "leduc_mccfr_final",
        "leduc_deep_cfr_early",
        "leduc_deep_cfr_intermediate",
        "leduc_deep_cfr_final",
        "modified_hulhe_random",
        "modified_hulhe_rule_based_v1",
        "modified_hulhe_deep_cfr_development",
    }
    assert "local_path" not in strategies["kuhn_cfr_final"]
    assert strategies["modified_hulhe_rule_based_v1"]["algorithm"] == "rule_based_v1"
    assert isinstance(registry.resolve("modified_hulhe_random").agent, BaselineAgent)
    assert isinstance(
        registry.resolve("modified_hulhe_rule_based_v1").agent,
        RuleBasedAgent,
    )


@pytest.mark.anyio
async def test_browser_assets_are_served_without_persistent_client_state() -> None:
    async with _client(_app(1)) as client:
        page = await client.get("/")
        script = await client.get("/static/app.js")
        stylesheet = await client.get("/static/styles.css")
        health = await client.get("/health")

    assert page.status_code == script.status_code == stylesheet.status_code == 200
    assert health.json() == {"status": "ok"}
    assert "game-select" in page.text
    assert "rules-dialog" in page.text
    assert "How does this game work?" in page.text
    assert "https://github.com/asbc0001/ac-cfr" in page.text
    assert "strategy-select" in page.text
    assert "policy-details" in page.text
    assert "net-result" in page.text
    assert "action-buttons" in page.text
    assert "opponent-cards" in page.text
    assert "localStorage" not in script.text
    assert "sessionStorage" not in script.text
    assert "/api/strategies" in script.text
    assert "expected_version" in script.text
    assert "human_utility" in script.text
    assert "completedHandIds" in script.text
    assert "Training iterations:" in script.text
    assert "Exact exploitability:" in script.text
    assert 'rank === "T" ? "10" : rank' in script.text
    assert "Choose a policy and start a hand" not in script.text
    assert "—" not in page.text
    assert "—" not in script.text


def test_registry_rejects_rule_agent_outside_modified_holdem(tmp_path: Path) -> None:
    registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    leduc_random = next(
        entry for entry in registry["strategies"] if entry["strategy_id"] == "leduc_random"
    )
    leduc_random["algorithm"] = "rule_based_v1"
    leduc_random["model_config_id"] = "rule_based_v1"
    path = tmp_path / "strategy_registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="rule-based.*modified HULHE"):
        load_strategy_registry(path, project_root=_PROJECT_ROOT)


@pytest.mark.parametrize(
    ("strategy_id", "private_card_count", "board_card_count", "bet_amount", "seed"),
    (
        ("kuhn_random", 1, 0, 1.0, 1),
        ("leduc_random", 1, 0, 2.0, 1),
        ("modified_hulhe_random", 2, 3, 1.0, 0),
        ("modified_hulhe_rule_based_v1", 2, 3, 1.0, 0),
    ),
)
@pytest.mark.anyio
async def test_start_hand_returns_only_player_visible_state(
    strategy_id: str,
    private_card_count: int,
    board_card_count: int,
    bet_amount: float,
    seed: int,
) -> None:
    async with _client(_app(seed)) as client:
        response = await client.post("/api/hands", json={"strategy_id": strategy_id})

    assert response.status_code == 201
    assert "set-cookie" not in response.headers
    hand = response.json()
    assert len(hand["hand_id"]) >= 32
    assert len(hand["private_cards"]) == private_card_count
    assert len(hand["board"]) == board_card_count
    if strategy_id == "leduc_random":
        assert hand["private_cards"][0][-1] in {"c", "d"}
    assert hand["opponent_cards"] == []
    assert hand["terminal_summary"] is None
    assert hand["current_hand"]
    assert "state" not in hand
    assert hand["legal_actions"]
    assert all(set(action) == {"action", "label", "amount"} for action in hand["legal_actions"])
    assert all(action["amount"] is None or action["amount"] > 0 for action in hand["legal_actions"])
    assert (
        next(action for action in hand["legal_actions"] if action["label"] == "Bet")["amount"]
        == bet_amount
    )
    assert not hand["terminal"]
    assert hand["human_player"] in {0, 1}
    assert hand["human_position"]


@pytest.mark.parametrize(
    "strategy_id",
    ("modified_hulhe_rule_based_v1", "leduc_random"),
)
@pytest.mark.anyio
async def test_inference_hand_advances_to_a_finite_terminal_result(strategy_id: str) -> None:
    async with _client(_app(4)) as client:
        hand, observed_ai_decision = await _play_passive_hand(client, strategy_id)

    assert observed_ai_decision
    assert hand["terminal"]
    assert hand["legal_actions"] == []
    assert len(hand["opponent_cards"]) == (2 if strategy_id.startswith("modified_hulhe") else 1)
    assert hand["terminal_summary"]["reason"] == "showdown"
    expected_highlights = 5 if strategy_id.startswith("modified_hulhe") else 1
    assert len(hand["terminal_summary"]["highlighted_cards"]) >= expected_highlights
    assert hand["terminal_summary"]["human_hand"]
    assert hand["terminal_summary"]["opponent_hand"]
    assert {entry["actor"] for entry in hand["action_history"]} <= {"You", "AI"}
    assert all(set(entry) == {"street", "actor", "action"} for entry in hand["action_history"])
    assert isinstance(hand["human_utility"], float)
    assert hand["result"]


@pytest.mark.anyio
async def test_temporary_holdem_neural_snapshot_plays_through_the_web_api(
    tmp_path: Path,
) -> None:
    registry_path = _write_holdem_neural_registry(tmp_path)
    app = create_app(
        registry_path=registry_path,
        project_root=tmp_path,
        master_rng=Random(4),
    )

    async with _client(app) as client:
        strategies = (await client.get("/api/strategies")).json()
        hand, observed_ai_decision = await _play_passive_hand(client, "holdem_neural_test")

    assert strategies[0]["strategy_id"] == "holdem_neural_test"
    assert strategies[0]["training_iteration"] == 1
    assert observed_ai_decision
    assert hand["terminal"]
    assert len(hand["opponent_cards"]) == 2
    assert hand["terminal_summary"]["reason"] == "showdown"
    assert len(hand["terminal_summary"]["highlighted_cards"]) >= 5


@pytest.mark.anyio
async def test_actions_reject_illegal_stale_unknown_and_extra_input_without_mutation() -> None:
    async with _client(_app(1)) as client:
        hand = (await client.post("/api/hands", json={"strategy_id": "kuhn_random"})).json()
        hand_path = f"/api/hands/{hand['hand_id']}/actions"
        version = hand["state_version"]
        assert {action["action"] for action in hand["legal_actions"]} == {1, 2}

        illegal = await client.post(
            hand_path,
            json={"expected_version": version, "action": 0},
        )
        assert illegal.status_code == 409

        accepted = await client.post(
            hand_path,
            json={"expected_version": version, "action": 1},
        )
        assert accepted.status_code == 200

        stale = await client.post(
            hand_path,
            json={"expected_version": version, "action": 1},
        )
        assert stale.status_code == 409
        malformed = await client.post(
            hand_path,
            json={
                "expected_version": accepted.json()["state_version"],
                "action": 1,
                "unexpected": True,
            },
        )
        assert malformed.status_code == 422
        missing = await client.post(
            "/api/hands/not-a-real-hand/actions",
            json={"expected_version": 0, "action": 1},
        )
        assert missing.status_code == 404


@pytest.mark.anyio
async def test_reset_discards_only_the_expected_hand_version() -> None:
    async with _client(_app(1)) as client:
        hand = (await client.post("/api/hands", json={"strategy_id": "kuhn_random"})).json()
        path = f"/api/hands/{hand['hand_id']}"
        stale = await client.delete(path, params={"expected_version": hand["state_version"] - 1})
        discarded = await client.delete(path, params={"expected_version": hand["state_version"]})
        repeated = await client.delete(path, params={"expected_version": hand["state_version"]})

    assert stale.status_code == 409
    assert discarded.status_code == 204
    assert repeated.status_code == 404


def test_abandoned_hand_expires() -> None:
    now = [0.0]
    registry = load_strategy_registry(_REGISTRY_PATH, project_root=_PROJECT_ROOT)
    store = HandStore(
        registry,
        hand_ttl_seconds=1.0,
        master_rng=Random(1),
        clock=lambda: now[0],
    )
    hand = store.create("kuhn_random")
    now[0] = 1.0

    with pytest.raises(HandNotFoundError, match="expired"):
        store.act(
            hand.hand_id,
            expected_version=hand.state_version,
            action=int(hand.legal_actions[0][0]),
        )


async def _play_passive_hand(
    client: httpx2.AsyncClient,
    strategy_id: str,
) -> tuple[dict[str, Any], bool]:
    response = await client.post("/api/hands", json={"strategy_id": strategy_id})
    assert response.status_code == 201
    hand = response.json()
    observed_ai_decision = False
    for _ in range(20):
        if hand["ai_decision"] is not None:
            observed_ai_decision = True
            probabilities = hand["ai_decision"]["probabilities"]
            assert sum(item["probability"] for item in probabilities) == pytest.approx(1.0)
            assert hand["ai_decision"]["chosen_action"] in {
                item["action"] for item in probabilities
            }
        if hand["terminal"]:
            break
        previous_version = hand["state_version"]
        passive_action = next(
            action for action in hand["legal_actions"] if action["label"] in {"Check", "Call"}
        )
        response = await client.post(
            f"/api/hands/{hand['hand_id']}/actions",
            json={
                "expected_version": previous_version,
                "action": passive_action["action"],
            },
        )
        assert response.status_code == 200
        hand = response.json()
        assert hand["state_version"] > previous_version
    return hand, observed_ai_decision


def _write_holdem_neural_registry(project_root: Path) -> Path:
    config = DeepCFRTrainingConfig(
        iterations=1,
        traversals_per_player=1,
        advantage_reservoir_capacity=1,
        strategy_reservoir_capacity=1,
        advantage_training_steps=1,
        strategy_training_steps=1,
        advantage_batch_size=1,
        strategy_batch_size=1,
        learning_rate=0.001,
        validation_fraction=0.1,
        max_gradient_norm=1.0,
        dropout_probability=0.0,
        seed=1,
        game_configuration_id=GameConfigurationId.MODIFIED_HULHE,
        model_config_id=ModelConfigId.MODIFIED_HULHE_DEEP_CFR,
        state_encoding_id=StateEncodingId.HOLD_EM,
    )
    artifact_path = project_root / "artifacts" / "holdem_neural_test.pt"
    export_deep_cfr_snapshot(
        artifact_path,
        network=build_deep_cfr_network(ModelConfigId.MODIFIED_HULHE_DEEP_CFR),
        game=HoldemConfig.modified(),
        config=config,
        implementation=DeepCFRImplementationId.OPTIMISED,
        snapshot_id="holdem_neural_test_iter_1",
        iteration=1,
        run_id="holdem_neural_test",
        source_checkpoint_id="test",
    )
    registry = {
        "schema_version": 1,
        "strategies": [
            {
                "strategy_id": "holdem_neural_test",
                "label": "Temporary Deep CFR",
                "game": "holdem",
                "game_version": "modified_hulhe",
                "algorithm": "deep_cfr",
                "agent_type": "neural",
                "snapshot_id": "holdem_neural_test_iter_1",
                "training_iteration": 1,
                "local_path": "artifacts/holdem_neural_test.pt",
                "evaluation": {},
                "model_config_id": "modified_hulhe_deep_cfr",
                "state_encoding": "holdem",
                "action_space": "poker",
                "tree_digest": "",
                "artifact_schema_version": 1,
                "release_id": "test",
                "file_size": artifact_path.stat().st_size,
                "sha256": file_sha256(artifact_path),
            }
        ],
    }
    metadata = load_deep_cfr_snapshot(artifact_path, HoldemConfig.modified()).metadata
    registry["strategies"][0]["tree_digest"] = metadata.tree_digest
    registry_path = project_root / "strategy_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path


def _client(app: FastAPI) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://testserver",
    )
