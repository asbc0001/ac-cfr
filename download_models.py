"""Checkout-local wrapper for the installed model-download command."""

from ac_cfr.cli.download_models import main

if __name__ == "__main__":
    raise SystemExit(main())
