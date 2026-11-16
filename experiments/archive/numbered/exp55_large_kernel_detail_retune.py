"""Compatibility entry point for targeted semantic-pole retuning."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.exp55_large_kernel_detail_retune", run_name="__main__")
else:
    from semantic_isp.exp55_large_kernel_detail_retune import *
