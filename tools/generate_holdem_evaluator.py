"""Generate deterministic production Hold'em evaluator lookup data."""

from pathlib import Path

from ac_cfr.games.holdem.evaluator.generation import write_lookup_data


def main() -> None:
    """Regenerate the packaged evaluator data"""
    repository_root = Path(__file__).resolve().parents[1]
    output_directory = repository_root / "src/ac_cfr/games/holdem/evaluator/data"
    metadata = write_lookup_data(output_directory)
    print(metadata["combined_sha256"])


if __name__ == "__main__":
    main()
