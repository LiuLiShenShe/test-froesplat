"""Independent, inference-only compression of frozen 2DGS PLY models."""
from .adapter import CompressionResult, compress_model

__all__ = ['CompressionResult', 'compress_model']
__version__ = '0.1.0'
