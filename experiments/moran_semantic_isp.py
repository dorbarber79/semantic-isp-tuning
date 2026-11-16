"""Compatibility imports for semantic projection and the linear-light ISP."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.moran_semantic_isp", run_name="__main__")
else:
    from semantic_isp.moran_semantic_isp import *
