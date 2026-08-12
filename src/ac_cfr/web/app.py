"""FastAPI application for ephemeral play against frozen poker strategies."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ac_cfr.persistence.registry import StrategyRegistryEntry, load_strategy_registry
from ac_cfr.web.gameplay import (
    DEFAULT_HAND_TTL_SECONDS,
    ActionHistoryEntry,
    HandNotFoundError,
    HandStore,
    InvalidHandActionError,
    PublicHandView,
    StaleHandVersionError,
    TerminalSummary,
)

DEFAULT_REGISTRY_PATH = Path("configs/strategy_registry.json")
STATIC_DIRECTORY = Path(__file__).with_name("static")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyResponse(_StrictModel):
    """Safe selector metadata for one registered playable strategy."""

    strategy_id: str
    label: str
    game: str
    game_version: str
    algorithm: str
    agent_type: str
    snapshot_id: str | None
    training_iteration: int
    evaluation: dict[str, object]


class StartHandRequest(_StrictModel):
    """Select one trusted registered strategy for a fresh hand."""

    strategy_id: Annotated[str, Field(min_length=1, max_length=128)]


class PlayActionRequest(_StrictModel):
    """One optimistic-concurrency-controlled human action."""

    expected_version: Annotated[int, Field(strict=True, ge=0)]
    action: Annotated[int, Field(strict=True, ge=0, le=2)]


class ActionResponse(_StrictModel):
    """One action currently available to the human player."""

    action: int
    label: str
    amount: float | None


class ActionProbabilityResponse(_StrictModel):
    """One AI action and its sampled-strategy probability."""

    action: int
    label: str
    probability: float


class AIDecisionResponse(_StrictModel):
    """The legal mixed strategy disclosed after the AI has acted."""

    probabilities: tuple[ActionProbabilityResponse, ...]
    chosen_action: int


class ActionHistoryResponse(_StrictModel):
    """One attributed action within a named betting round."""

    street: str
    actor: str
    action: str


class TerminalSummaryResponse(_StrictModel):
    """Player-safe explanation of a terminal fold or showdown."""

    reason: str
    headline: str
    human_hand: str | None
    opponent_hand: str | None
    highlighted_cards: tuple[str, ...]


class HandResponse(_StrictModel):
    """One complete player-visible snapshot of an ephemeral hand."""

    hand_id: str
    state_version: int
    strategy_id: str
    game: str
    game_version: str
    human_player: int
    human_position: str
    current_hand: str
    private_cards: tuple[str, ...]
    opponent_cards: tuple[str, ...]
    board: tuple[str, ...]
    pot: float
    action_history: tuple[ActionHistoryResponse, ...]
    legal_actions: tuple[ActionResponse, ...]
    ai_decision: AIDecisionResponse | None
    terminal: bool
    human_utility: float | None
    result: str | None
    terminal_summary: TerminalSummaryResponse | None


def create_app(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    project_root: Path | None = None,
    hand_ttl_seconds: float = DEFAULT_HAND_TTL_SECONDS,
    master_rng: Random | None = None,
) -> FastAPI:
    """Build one single-process application from a validated trusted registry."""
    root = Path.cwd() if project_root is None else project_root
    registry = load_strategy_registry(registry_path, project_root=root)
    store = HandStore(
        registry,
        hand_ttl_seconds=hand_ttl_seconds,
        master_rng=master_rng,
    )
    # Cache the three immutable assets so requests never depend on runtime file access.
    static_assets = {
        "index.html": (STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8"),
        "styles.css": (STATIC_DIRECTORY / "styles.css").read_text(encoding="utf-8"),
        "app.js": (STATIC_DIRECTORY / "app.js").read_text(encoding="utf-8"),
    }
    app = FastAPI(title="AC CFR poker demo", version="0.1.0")

    @app.get("/", response_class=Response, include_in_schema=False)
    async def index() -> Response:
        """Serve the single-page browser interface."""
        return Response(static_assets["index.html"], media_type="text/html")

    @app.get("/static/styles.css", response_class=Response, include_in_schema=False)
    async def stylesheet() -> Response:
        """Serve the immutable browser stylesheet."""
        return Response(static_assets["styles.css"], media_type="text/css")

    @app.get("/static/app.js", response_class=Response, include_in_schema=False)
    async def script() -> Response:
        """Serve the immutable browser application."""
        return Response(static_assets["app.js"], media_type="text/javascript")

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Report that the process and validated registry loaded successfully."""
        return {"status": "ok"}

    @app.get("/api/strategies", response_model=tuple[StrategyResponse, ...])
    async def list_strategies() -> tuple[StrategyResponse, ...]:
        """Return the registry-defined game/agent/snapshot matrix."""
        return tuple(_strategy_response(entry) for entry in registry.entries)

    @app.post(
        "/api/hands",
        response_model=HandResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def start_hand(request: StartHandRequest) -> HandResponse:
        """Start and automatically advance one new server-owned hand."""
        try:
            view = store.create(request.strategy_id)
        except (OSError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="strategy is unknown, unavailable, or incompatible",
            ) from error
        return _hand_response(view)

    @app.post("/api/hands/{hand_id}/actions", response_model=HandResponse)
    async def play_action(hand_id: str, request: PlayActionRequest) -> HandResponse:
        """Apply exactly one legal action to the expected hand version."""
        try:
            view = store.act(
                hand_id,
                expected_version=request.expected_version,
                action=request.action,
            )
        except HandNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="hand does not exist or has expired",
            ) from error
        except StaleHandVersionError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="hand state version is stale",
            ) from error
        except InvalidHandActionError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        return _hand_response(view)

    @app.delete("/api/hands/{hand_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def discard_hand(
        hand_id: str,
        expected_version: Annotated[int, Query(ge=0)],
    ) -> None:
        """Discard one exact hand version during an explicit browser reset."""
        try:
            store.discard(hand_id, expected_version=expected_version)
        except HandNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="hand does not exist or has expired",
            ) from error
        except StaleHandVersionError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="hand state version is stale",
            ) from error

    return app


def main(argv: Sequence[str] | None = None) -> None:
    """Run the initial single-worker Uvicorn development server."""
    parser = argparse.ArgumentParser(description="Serve the AC CFR poker demo API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--strategy-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--hand-ttl-seconds", type=float, default=DEFAULT_HAND_TTL_SECONDS)
    parsed = parser.parse_args(argv)
    app = create_app(
        registry_path=parsed.strategy_registry,
        project_root=parsed.project_root,
        hand_ttl_seconds=parsed.hand_ttl_seconds,
    )
    uvicorn.run(app, host=parsed.host, port=parsed.port, workers=1)


def _strategy_response(entry: StrategyRegistryEntry) -> StrategyResponse:
    return StrategyResponse(
        strategy_id=entry.strategy_id,
        label=entry.label,
        game=entry.game,
        game_version=entry.game_version,
        algorithm=entry.algorithm,
        agent_type=entry.agent_type,
        snapshot_id=entry.snapshot_id,
        training_iteration=entry.training_iteration,
        evaluation=entry.evaluation,
    )


def _hand_response(view: PublicHandView) -> HandResponse:
    ai_decision = view.ai_decision
    return HandResponse(
        hand_id=view.hand_id,
        state_version=view.state_version,
        strategy_id=view.strategy_id,
        game=view.game,
        game_version=view.game_version,
        human_player=view.human_player,
        human_position=view.human_position,
        current_hand=view.current_hand,
        private_cards=view.private_cards,
        opponent_cards=view.opponent_cards,
        board=view.board,
        pot=view.pot,
        action_history=tuple(_history_response(entry) for entry in view.action_history),
        legal_actions=tuple(
            ActionResponse(action=int(action), label=label, amount=amount)
            for action, label, amount in view.legal_actions
        ),
        ai_decision=(
            AIDecisionResponse(
                probabilities=tuple(
                    ActionProbabilityResponse(
                        action=int(probability.action),
                        label=probability.label,
                        probability=probability.probability,
                    )
                    for probability in ai_decision.probabilities
                ),
                chosen_action=int(ai_decision.chosen_action),
            )
            if ai_decision is not None
            else None
        ),
        terminal=view.terminal,
        human_utility=view.human_utility,
        result=view.result,
        terminal_summary=(
            _terminal_summary_response(view.terminal_summary)
            if view.terminal_summary is not None
            else None
        ),
    )


def _history_response(entry: ActionHistoryEntry) -> ActionHistoryResponse:
    return ActionHistoryResponse(
        street=entry.street,
        actor=entry.actor,
        action=entry.action,
    )


def _terminal_summary_response(summary: TerminalSummary) -> TerminalSummaryResponse:
    return TerminalSummaryResponse(
        reason=summary.reason,
        headline=summary.headline,
        human_hand=summary.human_hand,
        opponent_hand=summary.opponent_hand,
        highlighted_cards=summary.highlighted_cards,
    )
