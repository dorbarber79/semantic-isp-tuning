"""Compatibility entry point for direct ISP calibration."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp49_direct_isp_comparison", run_name="__main__")
else:
    from semantic_isp.exp49_direct_isp_comparison import *
