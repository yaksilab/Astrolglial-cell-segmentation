"""
Legacy pipeline entry point for backward compatibility.
This module provides the old import path for the pipeline.
"""

from .__main__ import main_pipeline

# For backward compatibility, allow direct import of main_pipeline
__all__ = ["main_pipeline"]
