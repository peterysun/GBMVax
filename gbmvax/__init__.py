"""
GBMVax — GBM-specific personalized cancer vaccine neoantigen pipeline.

Public surface:
    from gbmvax import load_config, __version__
"""

__version__ = "0.1.0"

from gbmvax.utils.config import load_config

__all__ = ["__version__", "load_config"]
