import hashlib
import json
from pathlib import Path

import pytest

from ac_cfr.cli.download_models import install_strategy_artifacts


def test_model_install_is_verified_atomic_and_idempotent(tmp_path: Path) -> None:
    contents = b"fixed test strategy contents"
    source_directory = tmp_path / "release"
    source_directory.mkdir()
    source_path = source_directory / "model.npz"
    source_path.write_bytes(contents)
    registry_path = tmp_path / "strategy_registry.json"
    registry_path.write_text(
        json.dumps(_registry_entry(contents)),
        encoding="utf-8",
    )

    installed = install_strategy_artifacts(
        registry_path=registry_path,
        project_root=tmp_path,
        source_directory=source_directory,
    )
    destination = tmp_path / "artifacts" / "model.npz"
    assert installed == (destination,)
    assert destination.read_bytes() == contents

    source_path.unlink()
    assert install_strategy_artifacts(
        registry_path=registry_path,
        project_root=tmp_path,
        source_directory=source_directory,
    ) == (destination,)

    destination.write_bytes(b"invalid")
    source_path.write_bytes(b"also invalid")
    with pytest.raises(ValueError, match="integrity"):
        install_strategy_artifacts(
            registry_path=registry_path,
            project_root=tmp_path,
            source_directory=source_directory,
        )
    assert destination.read_bytes() == b"invalid"
    assert not tuple(destination.parent.glob("*.tmp"))


def _registry_entry(contents: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "strategies": [
            {
                "strategy_id": "kuhn_test",
                "label": "Kuhn test",
                "game": "kuhn",
                "game_version": "kuhn",
                "algorithm": "cfr",
                "agent_type": "tabular",
                "snapshot_id": "kuhn_test_iter_1",
                "training_iteration": 1,
                "local_path": "artifacts/model.npz",
                "evaluation": {},
                "model_config_id": "tabular_average_strategy",
                "state_encoding": "kuhn",
                "action_space": "poker",
                "tree_digest": "0" * 64,
                "artifact_schema_version": 1,
                "release_id": "test-release",
                "file_size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        ],
    }
