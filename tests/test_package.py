from importlib.metadata import distribution, version

import ac_cfr


def test_installed_package_and_console_commands_are_available() -> None:
    assert ac_cfr.__package__ == "ac_cfr"
    assert version("ac-cfr") == "0.1.0"
    commands = {
        entry_point.name: entry_point.value
        for entry_point in distribution("ac-cfr").entry_points
        if entry_point.group == "console_scripts" and entry_point.name.startswith("ac-cfr-")
    }
    assert commands == {
        "ac-cfr-benchmark": "ac_cfr.cli.benchmark:main",
        "ac-cfr-evaluate": "ac_cfr.cli.evaluate:main",
        "ac-cfr-plot-results": "ac_cfr.cli.plot_results:main",
        "ac-cfr-train": "ac_cfr.cli.train:main",
    }
