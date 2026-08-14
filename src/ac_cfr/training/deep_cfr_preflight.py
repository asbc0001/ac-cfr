"""Fast provider-independent resource and execution checks before Deep CFR training."""

import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ac_cfr.common.config import GameConfigurationId
from ac_cfr.common.environment import effective_memory_bytes, environment_record
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.models import build_deep_cfr_network, deep_cfr_network_config
from ac_cfr.solvers.deep_cfr import DeepCFR
from ac_cfr.solvers.deep_cfr_selection import deep_cfr_solver_type
from ac_cfr.training.config import DeepCFRRuntimeConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

_ITERATION_BYTES = 4
_CHECKPOINT_ITERATION_WIDENING_BYTES = 4


def preflight_deep_cfr(config: DeepCFRRunConfig, runs_root: Path) -> dict[str, object]:
    """Check paths, resources, device execution, and one tiny traversal without training."""
    if not isinstance(config, DeepCFRRunConfig):
        raise TypeError("config must be a DeepCFRRunConfig")
    runs_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".deep_cfr_preflight_", dir=runs_root):
        pass

    estimates = _resource_estimates(config)
    available_memory, _ = effective_memory_bytes()
    filesystem_free_disk = shutil.disk_usage(runs_root).free
    storage_budget = config.runtime.storage_budget_bytes
    free_disk = (
        filesystem_free_disk
        if storage_budget is None
        else min(filesystem_free_disk, storage_budget)
    )
    if available_memory < estimates["minimum_available_memory_bytes"]:
        raise OSError(
            "insufficient available memory for configured reservoirs: "
            f"need at least {estimates['minimum_available_memory_bytes']:,} bytes, "
            f"found {available_memory:,}"
        )
    if free_disk < estimates["minimum_free_disk_bytes"]:
        raise OSError(
            "insufficient free storage for retained atomic checkpoints: "
            f"need at least {estimates['minimum_free_disk_bytes']:,} bytes, "
            f"found {free_disk:,}"
        )
    torch.set_num_threads(config.runtime.cpu_threads)
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")

    _check_network_update(config)
    traversal_samples = _check_one_traversal(config)
    return {
        "status": "passed",
        "run_config": config.to_dict(),
        "environment": environment_record(
            "torch",
            "numpy",
            "psutil",
            device=config.runtime.device,
        ),
        "resources": {
            **estimates,
            "available_memory_bytes": available_memory,
            "filesystem_free_disk_bytes": filesystem_free_disk,
            "storage_budget_bytes": storage_budget,
            "free_disk_bytes": free_disk,
        },
        "checks": {
            "runs_root_writable": True,
            "network_forward_backward_finite": True,
            "checked_training_batch_size": config.training.advantage_batch_size,
            "tiny_traversal_completed": True,
            "tiny_traversal_advantage_samples": traversal_samples,
        },
    }


def _resource_estimates(config: DeepCFRRunConfig) -> dict[str, int]:
    """Estimate configured reservoir RAM and bounded checkpoint storage."""
    network_config = deep_cfr_network_config(
        config.training.model_config_id,
        dropout_probability=config.training.dropout_probability,
    )
    action_count = network_config.output_size
    state_item_bytes = (
        2 if config.training.game_configuration_id is GameConfigurationId.MODIFIED_HULHE else 4
    )
    state_bytes = network_config.input_size * state_item_bytes
    sampling_weight_bytes = 4 if config.training.opponent_exploration_epsilon > 0.0 else 0
    common_sample_bytes = state_bytes + action_count + _ITERATION_BYTES + sampling_weight_bytes
    advantage_sample_bytes = common_sample_bytes + action_count * 4
    strategy_sample_bytes = advantage_sample_bytes + 1
    reservoir_bytes = (
        2 * config.training.advantage_reservoir_capacity * advantage_sample_bytes
        + config.training.strategy_reservoir_capacity * strategy_sample_bytes
    )
    retained_samples = (
        2 * config.training.advantage_reservoir_capacity
        + config.training.strategy_reservoir_capacity
    )
    checkpoint_bytes = reservoir_bytes + retained_samples * _CHECKPOINT_ITERATION_WIDENING_BYTES
    minimum_memory = int(reservoir_bytes * 1.25) + 1024**3
    minimum_disk = int(checkpoint_bytes * (config.checkpoint_retention + 1) * 1.1) + 1024**3
    return {
        "configured_reservoir_bytes": reservoir_bytes,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "minimum_available_memory_bytes": minimum_memory,
        "minimum_free_disk_bytes": minimum_disk,
    }


def _check_network_update(config: DeepCFRRunConfig) -> None:
    """Run one finite optimiser update on the configured device and architecture."""
    device = torch.device(config.runtime.device)
    network = build_deep_cfr_network(
        config.training.model_config_id,
        dropout_probability=config.training.dropout_probability,
    ).to(device)
    batch_size = config.training.advantage_batch_size
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    states = torch.randn(
        batch_size,
        network.config.input_size,
        generator=generator,
        device=device,
    )
    targets = torch.zeros(batch_size, network.config.output_size, device=device)
    optimiser = torch.optim.Adam(network.parameters(), lr=config.training.learning_rate)
    optimiser.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(network(states), targets)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("preflight network loss is non-finite")
    loss.backward()
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in network.parameters()
    ):
        raise FloatingPointError("preflight network gradients are non-finite")
    optimiser.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
        raise FloatingPointError("preflight network parameters are non-finite")
    del optimiser, network, states, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _check_one_traversal(config: DeepCFRRunConfig) -> int:
    """Exercise one configured game traversal with tiny temporary reservoirs."""
    training = replace(
        config.training,
        iterations=1,
        traversals_per_player=1,
        advantage_reservoir_capacity=128,
        strategy_reservoir_capacity=128,
        advantage_training_steps=1,
        strategy_training_steps=1,
        advantage_batch_size=4,
        strategy_batch_size=4,
        snapshot_iterations=(),
    )
    runtime = DeepCFRRuntimeConfig(
        inference_batch_size=min(4, config.runtime.inference_batch_size),
        cpu_threads=config.runtime.cpu_threads,
        device=config.runtime.device,
        traversal_workers=1,
    )
    if training.game_configuration_id is GameConfigurationId.MODIFIED_HULHE:
        solver = DeepCFR(HoldemConfig.modified(), training, runtime)
        solver.collect_calibration_traversals(0)
    else:
        tree = compile_game_tree(LeducGame(), LeducConfig())
        solver = deep_cfr_solver_type(config.implementation)(tree, training, runtime)
        solver.train(1)
    return len(solver.advantage_reservoirs[0])
