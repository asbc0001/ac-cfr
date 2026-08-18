"use strict";

const elements = {
  game: document.querySelector("#game-select"),
  opponent: document.querySelector("#opponent-select"),
  opponentDetails: document.querySelector("#opponent-details"),
  rulesButton: document.querySelector("#rules-button"),
  rulesDialog: document.querySelector("#rules-dialog"),
  rulesTitle: document.querySelector("#rules-title"),
  rulesSummary: document.querySelector("#rules-summary"),
  rulesList: document.querySelector("#rules-list"),
  start: document.querySelector("#start-button"),
  status: document.querySelector("#status-message"),
  table: document.querySelector("#table"),
  heading: document.querySelector("#table-heading"),
  turnStatus: document.querySelector("#turn-status"),
  netResult: document.querySelector("#net-result"),
  humanPosition: document.querySelector("#human-position"),
  aiPosition: document.querySelector("#ai-position"),
  pot: document.querySelector("#pot"),
  privateCards: document.querySelector("#private-cards"),
  currentHandLabel: document.querySelector("#current-hand"),
  boardCards: document.querySelector("#board-cards"),
  opponentCards: document.querySelector("#opponent-cards"),
  opponentHandLabel: document.querySelector("#opponent-hand"),
  actions: document.querySelector("#action-buttons"),
  result: document.querySelector("#result"),
  showdownDetails: document.querySelector("#showdown-details"),
  aiDecision: document.querySelector("#ai-decision"),
  history: document.querySelector("#action-history"),
};

const gameLabels = {
  kuhn: "Kuhn poker",
  leduc: "Leduc hold’em",
  holdem: "Modified HULHE",
};

const algorithmLabels = {
  uniform_random: "Random",
  rule_based_v1: "Rule-based",
  cfr: "CFR",
  cfr_plus: "CFR+",
  mccfr: "MCCFR",
  deep_cfr: "Deep CFR",
};

const gameRules = {
  kuhn: {
    title: "Kuhn poker",
    summary: "A minimal poker game with one private card and one betting round.",
    points: [
      "The deck contains J, Q and K. Each player receives one card and contributes one chip.",
      "Players may check or bet. A bet can be called or folded to.",
      "If nobody folds, the higher card wins the pot.",
    ],
  },
  leduc: {
    title: "Leduc hold’em",
    summary: "A compact two-round game with private and public cards.",
    points: [
      "The deck contains two suits each of J, Q and K. Each player receives one private card.",
      "One public card is dealt after the first betting round, followed by a second round.",
      "A pair wins over a high card. Otherwise, the higher private card wins.",
      "Bets cost 2 chips in round one and 4 chips in round two, with one raise allowed.",
    ],
  },
  holdem: {
    title: "Modified HULHE",
    summary: "A shorter fixed-limit hold’em variant that starts on the flop.",
    points: [
      "Each player receives two private cards, and the flop starts face up.",
      "Betting occurs on the flop, turn and river. The button acts last on each street.",
      "Bets cost 1 chip on the flop and 2 chips on the turn and river, with one raise allowed.",
      "At showdown, the best five-card poker hand made from private and board cards wins.",
    ],
  },
};

let strategies = [];
let currentHand = null;
let requestPending = false;
// Page-memory state survives New hand without creating persistent browser data.
let trackedStrategyId = null;
let netResult = 0;
let completedHandCount = 0;
const completedHandIds = new Set();

function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function unique(values) {
  return [...new Set(values)];
}

function setOptions(select, values, labelFor) {
  select.replaceChildren(...values.map((value) => option(value, labelFor(value))));
  select.disabled = values.length === 0;
}

function updateOpponents() {
  const available = strategies.filter((entry) => entry.game === elements.game.value);
  setOptions(
    elements.opponent,
    available.map((entry) => entry.strategy_id),
    (strategyId) => available.find((entry) => entry.strategy_id === strategyId).label,
  );
  elements.rulesButton.disabled = strategies.length === 0;
  elements.start.disabled = requestPending || available.length === 0;
  renderOpponentDetails();
}

function formatMetric(value) {
  return value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function formatHeadToHeadEvaluation(evaluation) {
  if (typeof evaluation.value !== "number") {
    return null;
  }
  const opponent = algorithmLabels[evaluation.opponent] ?? evaluation.opponent;
  const label = typeof opponent === "string" ? `Evaluation vs ${opponent}` : "Evaluation";
  return `${label}: ${formatMetric(evaluation.value)} ${evaluation.utility_unit}`;
}

function renderNetResult() {
  const sign = netResult > 0 ? "+" : "";
  const handLabel = completedHandCount === 1 ? "hand" : "hands";
  elements.netResult.textContent =
    `Net: ${sign}${formatMetric(netResult)} chips · ${completedHandCount} ${handLabel}`;
}

function resetNetResult() {
  trackedStrategyId = elements.opponent.value || null;
  netResult = 0;
  completedHandCount = 0;
  completedHandIds.clear();
  renderNetResult();
}

function recordTerminalResult() {
  if (
    !currentHand.terminal ||
    currentHand.human_utility === null ||
    currentHand.strategy_id !== trackedStrategyId ||
    completedHandIds.has(currentHand.hand_id)
  ) {
    return;
  }
  completedHandIds.add(currentHand.hand_id);
  netResult += currentHand.human_utility;
  completedHandCount += 1;
  renderNetResult();
}

function renderOpponentDetails() {
  const selected = strategies.find((entry) => entry.strategy_id === elements.opponent.value);
  if (selected === undefined) {
    elements.opponentDetails.textContent = "";
    return;
  }
  if (selected.training_iteration > 0) {
    const details = [`Training iterations: ${selected.training_iteration.toLocaleString()}`];
    if (typeof selected.evaluation.exploitability === "number") {
      details.push(`Exact exploitability: ${formatMetric(selected.evaluation.exploitability)} chips`);
    } else {
      const evaluation = formatHeadToHeadEvaluation(selected.evaluation);
      if (evaluation !== null) {
        details.push(evaluation);
      }
    }
    elements.opponentDetails.textContent = details.join(" · ");
    return;
  }
  if (selected.algorithm === "rule_based_v1") {
    const evaluation = formatHeadToHeadEvaluation(selected.evaluation);
    const result = evaluation === null ? "" : ` · ${evaluation}`;
    elements.opponentDetails.textContent = `Fixed rule-based baseline${result}`;
    return;
  }
  elements.opponentDetails.textContent = "Uniform random baseline · no training iterations";
}

function setPending(pending) {
  requestPending = pending;
  elements.game.disabled = pending || strategies.length === 0;
  elements.opponent.disabled = pending || elements.opponent.options.length === 0;
  elements.start.disabled = pending || elements.opponent.options.length === 0;
  elements.actions.querySelectorAll("button").forEach((button) => {
    button.disabled = pending;
  });
  if (pending && currentHand !== null && !currentHand.terminal) {
    elements.turnStatus.textContent = "AI acting…";
  }
}

function showStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Retain the status-based message when the response has no JSON body.
    }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

async function discardCurrentHand() {
  if (currentHand === null) {
    return;
  }
  const hand = currentHand;
  currentHand = null;
  try {
    await apiRequest(
      `/api/hands/${encodeURIComponent(hand.hand_id)}?expected_version=${hand.state_version}`,
      { method: "DELETE" },
    );
  } catch {
    // Reset remains local even if the old hand expired or was already discarded.
  }
}

async function startHand() {
  setPending(true);
  showStatus("Dealing a new hand…");
  try {
    await discardCurrentHand();
    currentHand = await apiRequest("/api/hands", {
      method: "POST",
      body: JSON.stringify({ strategy_id: elements.opponent.value }),
    });
    if (trackedStrategyId !== currentHand.strategy_id) {
      resetNetResult();
    }
    renderHand();
    showStatus("");
  } catch (error) {
    elements.table.hidden = true;
    showStatus(error.message, true);
  } finally {
    setPending(false);
  }
}

async function playAction(action) {
  if (currentHand === null || requestPending) {
    return;
  }
  setPending(true);
  showStatus("Applying action…");
  try {
    currentHand = await apiRequest(
      `/api/hands/${encodeURIComponent(currentHand.hand_id)}/actions`,
      {
        method: "POST",
        body: JSON.stringify({
          expected_version: currentHand.state_version,
          action,
        }),
      },
    );
    renderHand();
    showStatus("");
  } catch (error) {
    showStatus(`${error.message}. Start a new hand if this page is out of date.`, true);
  } finally {
    setPending(false);
  }
}

function showGameRules() {
  const rules = gameRules[elements.game.value];
  if (rules === undefined) {
    return;
  }
  elements.rulesTitle.textContent = rules.title;
  elements.rulesSummary.textContent = rules.summary;
  elements.rulesList.replaceChildren(
    ...rules.points.map((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      return item;
    }),
  );
  elements.rulesDialog.showModal();
}

function renderCards(
  container,
  cards,
  emptyCount,
  placeholderClass = "empty-card",
  placeholderLabels = [],
) {
  const highlightedCards = new Set(currentHand.terminal_summary?.highlighted_cards ?? []);
  const showHighlights = currentHand.terminal_summary?.reason === "showdown";
  const visible = cards.map((card) => {
    const item = document.createElement("span");
    item.className = `card${/[hd]$/i.test(card) ? " red" : ""}`;
    if (showHighlights) {
      item.classList.add(highlightedCards.has(card) ? "winning-card" : "unused-card");
    }
    const suits = { c: "♣", d: "♦", h: "♥", s: "♠" };
    const suit = suits[card.at(-1)];
    const rank = card.slice(0, -1);
    const displayRank = rank === "T" ? "10" : rank;
    item.textContent = suit === undefined ? card : `${displayRank}${suit}`;
    return item;
  });
  const placeholders = Array.from(
    { length: Math.max(0, emptyCount - cards.length) },
    (_, offset) => {
      const item = document.createElement("span");
      item.className = placeholderClass;
      const slotIndex = cards.length + offset;
      if (placeholderClass === "hidden-card") {
        item.textContent = "AC";
        item.setAttribute("aria-label", "Hidden AI card");
      } else {
        item.textContent = placeholderLabels[slotIndex] ?? "";
        item.setAttribute("aria-hidden", "true");
      }
      return item;
    },
  );
  container.replaceChildren(...visible, ...placeholders);
}

function renderActions() {
  const buttons = currentHand.legal_actions.map((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action.amount === null ? action.label : `${action.label} ${action.amount}`;
    button.addEventListener("click", () => playAction(action.action));
    return button;
  });
  elements.actions.replaceChildren(...buttons);
}

function renderAIDecision() {
  if (currentHand.ai_decision === null) {
    elements.aiDecision.className = "empty-copy";
    elements.aiDecision.textContent = "The AI has not acted yet.";
    return;
  }
  const rows = currentHand.ai_decision.probabilities.map((entry) => {
    const percentage = entry.probability * 100;
    const row = document.createElement("div");
    row.className = "probability-row";
    const label = document.createElement("span");
    label.textContent = entry.label;
    if (entry.action === currentHand.ai_decision.chosen_action) {
      label.className = "chosen-probability";
    }
    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.className = "probability-fill";
    fill.style.width = `${percentage}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.textContent = `${percentage.toFixed(1)}%`;
    row.append(label, track, value);
    return row;
  });
  const list = document.createElement("div");
  list.className = "probability-list";
  list.append(...rows);
  elements.aiDecision.className = "";
  elements.aiDecision.replaceChildren(list);
}

function renderHistory() {
  let previousStreet = null;
  const entries = currentHand.action_history.map((entry) => {
    const item = document.createElement("li");
    if (entry.street !== previousStreet) {
      const street = document.createElement("strong");
      street.textContent = entry.street;
      item.append(street);
      item.classList.add("street-start");
      previousStreet = entry.street;
    }
    item.append(`${entry.actor}: ${entry.action}`);
    return item;
  });
  if (entries.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No player actions yet";
    entries.push(item);
  }
  elements.history.replaceChildren(...entries);
}

function renderTerminalSummary() {
  const summary = currentHand.terminal_summary;
  if (summary === null) {
    elements.result.textContent = "";
    elements.showdownDetails.hidden = true;
    elements.showdownDetails.replaceChildren();
    return;
  }
  elements.result.textContent = summary.headline;
  const details = [];
  if (summary.human_hand !== null) {
    details.push(`Your hand: ${summary.human_hand}`);
  }
  if (summary.opponent_hand !== null) {
    details.push(`AI hand: ${summary.opponent_hand}`);
  }
  if (currentHand.result !== null) {
    details.push(`Net result: ${currentHand.result}`);
  }
  elements.showdownDetails.replaceChildren(
    ...details.map((text) => {
      const line = document.createElement("p");
      line.textContent = text;
      return line;
    }),
  );
  elements.showdownDetails.hidden = details.length === 0;
}

function renderPositions() {
  if (currentHand.game === "holdem") {
    const humanHasButton = currentHand.human_position === "Button";
    elements.humanPosition.textContent = humanHasButton ? "BTN" : "";
    elements.aiPosition.textContent = humanHasButton ? "" : "BTN";
    return;
  }
  elements.humanPosition.textContent = `P${currentHand.human_player + 1}`;
  elements.aiPosition.textContent = `P${2 - currentHand.human_player}`;
}

function renderHand() {
  const selected = strategies.find((entry) => entry.strategy_id === currentHand.strategy_id);
  elements.table.hidden = false;
  const gameLabel = gameLabels[currentHand.game] ?? currentHand.game;
  elements.heading.textContent = `${gameLabel} vs ${selected.label}`;
  elements.turnStatus.textContent = currentHand.terminal ? "Hand complete" : "Your turn";
  elements.pot.textContent = `Pot: ${currentHand.pot.toLocaleString()} chips`;
  renderPositions();
  const privateCardSlots = currentHand.game === "holdem" ? 2 : 1;
  renderCards(elements.privateCards, currentHand.private_cards, privateCardSlots);
  elements.currentHandLabel.textContent = currentHand.current_hand;
  const boardSlots = { kuhn: 0, leduc: 1, holdem: 5 };
  const boardLabels = {
    leduc: ["Public"],
    holdem: ["", "", "", "Turn", "River"],
  };
  renderCards(
    elements.boardCards,
    currentHand.board,
    boardSlots[currentHand.game] ?? 0,
    "empty-card",
    boardLabels[currentHand.game] ?? [],
  );
  renderCards(
    elements.opponentCards,
    currentHand.opponent_cards,
    privateCardSlots,
    "hidden-card",
  );
  const opponentHand = currentHand.terminal_summary?.opponent_hand ?? null;
  elements.opponentHandLabel.textContent = opponentHand ?? "";
  elements.opponentHandLabel.hidden = opponentHand === null;
  renderActions();
  renderAIDecision();
  renderHistory();
  renderTerminalSummary();
  recordTerminalResult();
}

function handleGameChange() {
  updateOpponents();
  resetNetResult();
}

function handleOpponentChange() {
  renderOpponentDetails();
  resetNetResult();
}

async function initialise() {
  try {
    strategies = await apiRequest("/api/strategies");
    const games = unique(strategies.map((entry) => entry.game));
    setOptions(elements.game, games, (value) => gameLabels[value] ?? value);
    updateOpponents();
    showStatus("");
  } catch (error) {
    showStatus(`Could not load playable policies: ${error.message}`, true);
  }
}

elements.game.addEventListener("change", handleGameChange);
elements.opponent.addEventListener("change", handleOpponentChange);
elements.rulesButton.addEventListener("click", showGameRules);
elements.start.addEventListener("click", startHand);
initialise();
