"""Compatibility entry point for semantic pole rendering."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("semantic_isp.render_pole_centroids", run_name="__main__")
else:
    from semantic_isp.render_pole_centroids import *
