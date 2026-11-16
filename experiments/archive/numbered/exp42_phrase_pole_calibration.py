"""Compatibility entry point for phrase-pole calibration."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp42_phrase_pole_calibration", run_name="__main__")
else:
    from semantic_isp.exp42_phrase_pole_calibration import *
