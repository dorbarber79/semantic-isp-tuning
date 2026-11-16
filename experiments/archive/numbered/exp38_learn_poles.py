"""Compatibility entry point for the active phrase optimizer."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp38_learn_poles", run_name="__main__")
else:
    from semantic_isp.exp38_learn_poles import *
