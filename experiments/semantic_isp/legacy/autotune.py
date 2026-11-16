"""Clear entry point for the current four-mode protection-aware autotune."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp69_protection_retune", run_name="__main__")
else:
    from semantic_isp.exp69_protection_retune import run

__all__ = ["run"]
