"""Shared default settings for the product-recognition workflow.

This module centralizes defaults used by CLI scripts and backend-facing APIs:
folder layout, model name, split policy, and output label. Runtime behavior
lives in the dedicated workflow modules.
"""

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_CACHE_DIR = "data/outputs/memory_bank_cache"
DEFAULT_OUTPUT_DIR = "data/outputs/threshold_optimization_results"
DEFAULT_UNKNOWN_LABEL = "__Unknown__"
DEFAULT_MIN_IMAGES_PER_CLASS = 20
DEFAULT_TEST_RATIO = 0.20
DEFAULT_SEED = 42
DEFAULT_MAX_IMAGES_PER_CLASS = 100
DEFAULT_TOP_K = 5
