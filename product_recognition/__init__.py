"""Public package exports for the product-recognition MVP.

Input/output details live on the exported functions. Most backend integrations
only need `load_recognizer` and `predict_image`.
"""

from .inference import load_recognizer, predict_embedding, predict_image, predict_images
from .memory_bank import load_memory_bank_cache, save_memory_bank_cache

__all__ = [
    "load_recognizer",
    "predict_embedding",
    "predict_image",
    "predict_images",
    "load_memory_bank_cache",
    "save_memory_bank_cache",
]
