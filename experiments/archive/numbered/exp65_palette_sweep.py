"""Compatibility entry point for the palette coordinate sweep."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp65_palette_sweep", run_name="__main__")
else:
    from semantic_isp.exp65_palette_sweep import *
