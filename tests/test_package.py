from importlib.metadata import version

import ac_cfr


def test_installed_package_is_importable() -> None:
    assert ac_cfr.__package__ == "ac_cfr"
    assert version("ac-cfr") == "0.1.0"
