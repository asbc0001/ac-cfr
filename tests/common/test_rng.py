import pytest

from ac_cfr.common.rng import RngStream, SeedDeriver


def test_named_seed_derivation_is_reproducible_and_independent() -> None:
    first = SeedDeriver(2026)
    second = SeedDeriver(2026)

    assert first.derive(RngStream.CHANCE) == second.derive(RngStream.CHANCE)
    assert first.derive(RngStream.CHANCE) != first.derive(RngStream.POLICY)
    assert first.derive(RngStream.WORKER, 0) != first.derive(RngStream.WORKER, 1)
    assert first.derive(RngStream.CHANCE) != SeedDeriver(2027).derive(RngStream.CHANCE)
    assert first.derive(RngStream.CHANCE) == 2160575435672598360
    assert first.derive(RngStream.WORKER, 7) == 8699089107685535998


def test_python_rng_state_can_be_restored_independently() -> None:
    seeds = SeedDeriver(42)
    chance_rng = seeds.python_rng(RngStream.CHANCE)
    policy_rng = seeds.python_rng(RngStream.POLICY)

    saved_state = chance_rng.getstate()
    chance_values = [chance_rng.random() for _ in range(3)]
    chance_rng.setstate(saved_state)

    assert chance_values == [chance_rng.random() for _ in range(3)]
    assert chance_rng.getstate() != policy_rng.getstate()


def test_seed_inputs_are_validated() -> None:
    for invalid_root_seed in (-1, 1 << 128):
        with pytest.raises(ValueError):
            SeedDeriver(invalid_root_seed)
    for invalid_root_seed in (True, 1.5, "1"):
        with pytest.raises(TypeError):
            SeedDeriver(invalid_root_seed)  # type: ignore[arg-type]

    seeds = SeedDeriver(0)
    with pytest.raises(TypeError):
        seeds.derive("chance")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        seeds.derive(RngStream.WORKER, -1)
    with pytest.raises(TypeError):
        seeds.derive(RngStream.WORKER, True)
