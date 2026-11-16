"""Compatibility entry point for relevance-gate calibration."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp47_relevance_calibration", run_name="__main__")
else:
    from semantic_isp.exp47_relevance_calibration import *
