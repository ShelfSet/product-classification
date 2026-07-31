# Product Recognition

## Overview

This project provides a memory-bank product recognizer for cropped product images.
It is designed for client integration where a backend loads the recognition
runtime once, keeps it in memory, and then classifies incoming crop images.

The system does not train a traditional classifier. Instead, it:

1. Embeds known product crop images with a vision embedding model.
2. Stores those embeddings in a memory-bank cache.
3. Uses FAISS nearest-neighbor search to compare a new crop against the memory bank.
4. Applies calibrated score thresholds to decide whether the best match is trusted.
5. Returns either a known product label or the unknown label, `__Unknown__`.

Known product labels use this format:

```text
brand__product
```

The normal workflow has three stages:

1. Create a memory-bank cache from a labeled crop dataset.
2. Optimize known/unknown thresholds from the cache.
3. Run inference on one crop or a folder of crops.

For deployment, the most important artifacts are:

- the memory-bank `.pkl` cache
- `thresholds.csv`
- the Python package under `product_recognition/`
- the requirements file matching the target environment

## Project Structure

```text
product-classification/
  product_recognition/
    cli/
      create_memory_bank.py
      optimize_thresholds.py
      run_inference.py
    __init__.py
    config.py
    dataset.py
    embeddings.py
    inference.py
    memory_bank.py
    thresholds.py
  data/
    dataset/
    datasetv6_crops/
    input_images/
    outputs/
    README.md
  README.md
  requirements.txt
  requirements-cuda118.txt
  requirements-cuda121.txt
  requirements-cuda128.txt
```

## Main Files

`product_recognition/config.py`

Centralizes default settings: model name, output folders, unknown label, split
settings, random seed, memory-bank image cap, and default `top_k`.

`product_recognition/dataset.py`

Scans a dataset arranged as `brand/product/images`. It separates known products
from unknown/negative examples and creates leakage-safe memory/test splits. The
split logic groups crops from the same source image so near-duplicate crops do
not appear in both memory and evaluation sets.

`product_recognition/embeddings.py`

Loads the embedding model and converts images into normalized `float32`
embeddings. All embedding extraction lives here so memory creation, threshold
optimization, and inference use the same preprocessing and normalization.

`product_recognition/memory_bank.py`

Builds and saves the memory-bank cache. The cache stores memory items,
known-test items, unknown/negative test items, optional precomputed embeddings,
and metadata. Runtime inference loads this cache before building the FAISS index.

`product_recognition/thresholds.py`

Optimizes rejection thresholds for open-set recognition. It selects one global
threshold and optional per-class thresholds, then writes:

- `thresholds.csv`: deployable threshold table
- `thresholds_w_metrics.csv`: threshold table with diagnostic metrics

`product_recognition/inference.py`

Backend-facing inference code. This module loads the recognizer runtime, builds
the FAISS index, embeds incoming images, aggregates nearest-neighbor scores, and
applies the threshold policy.

Most backend integrations only need:

- `load_recognizer(...)`
- `predict_image(...)`
- `predict_images(...)`

`product_recognition/__init__.py`

Exports the main package functions for Python imports.

`product_recognition/cli/create_memory_bank.py`

Command-line entry point for scanning a dataset, creating memory/test splits,
embedding memory images, and saving the memory-bank cache.

`product_recognition/cli/optimize_thresholds.py`

Command-line entry point for loading a memory-bank cache, scoring known and
unknown evaluation images, and writing threshold artifacts.

`product_recognition/cli/run_inference.py`

Command-line entry point for classifying one image or a folder of images.
When `--output-csv` is provided, it writes both compact and detailed CSV outputs.

`data/README.md`

Documents the intended local data folder layout.

`data/dataset/`

Small placeholder dataset structure. Use this as a template for the required
`brand/product/image` layout.

`data/datasetv6_crops/`

Larger crop dataset folder, if included in the handover. This is useful for
rebuilding the memory bank and thresholds, but it is not required for runtime if
the final cache and threshold files are already provided.

`data/input_images/`

Example crop images for inference testing.

`data/outputs/memory_bank_cache/`

Stores generated memory-bank `.pkl` files.

`data/outputs/threshold_optimization_results/`

Stores threshold optimization outputs:

- `thresholds.csv`
- `thresholds_w_metrics.csv`

`data/outputs/inference_results/`

Stores inference CSV outputs:

- `inference_results.csv`
- `inference_results_detailed.csv`

`requirements.txt`

Default CPU installation dependencies.

`requirements-cuda118.txt`, `requirements-cuda121.txt`, `requirements-cuda128.txt`

CUDA-specific dependency files. Use the one that matches the target machine's
CUDA/PyTorch setup.

## Environment Setup

Run all commands from the repository root.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, either allow local script execution for the
session or use the Command Prompt activation script:

```bat
.venv\Scripts\activate.bat
```

Install the default CPU dependencies:

```powershell
pip install -r requirements.txt
```

For a CUDA machine, install the matching CUDA requirements instead:

```powershell
pip install -r requirements-cuda118.txt
```

```powershell
pip install -r requirements-cuda121.txt
```

```powershell
pip install -r requirements-cuda128.txt
```

Check whether PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If CUDA should be available but prints `False`, reinstall the CUDA requirements
inside the active virtual environment:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --force-reinstall --no-cache-dir -r requirements-cuda128.txt
```

The default embedding model is `Qwen/Qwen3-VL-Embedding-2B`. The first model load
may download model weights unless they are already available in the local model
cache or provided from a local model path.

## Dataset Layout

The dataset scanner expects this folder structure:

```text
dataset_root/
  Brand A/
    Product A/
      image_1.jpg
      image_2.jpg
  Brand B/
    Product B/
      image_1.jpg
      image_2.jpg
  __Unknown__/
    unknown/
      negative_1.jpg
      negative_2.jpg
```

Known products are read from `Brand/Product` folders and become labels like:

```text
Brand A__Product A
```

Unknown images are used only for threshold calibration. They are not added to the
memory bank as a product class.

Supported image extensions are:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

## CLI Usage

Use one cache name consistently across all commands. The included handover
artifacts use:

```text
product_memory_bank_real_dataset
```

If your dataset root is different, replace `data/datasetv6_crops` and choose a
cache name that clearly identifies the dataset/version.

### 1. Create the Memory-Bank Cache

```powershell
python -m product_recognition.cli.create_memory_bank --dataset-root data/datasetv6_crops --cache-dir data/outputs/memory_bank_cache --cache-name product_memory_bank_real_dataset
```

This command:

- scans the dataset
- removes classes with too few images
- creates leakage-safe memory and known-test splits
- embeds the memory images
- writes a `.pkl` cache under `data/outputs/memory_bank_cache/`

Useful optional arguments:

```text
--min-images-per-class 20
--test-ratio 0.20
--seed 42
--max-images-per-class 100
--device auto
--batch-size 16
```

### 2. Optimize Thresholds

```powershell
python -m product_recognition.cli.optimize_thresholds --cache-dir data/outputs/memory_bank_cache --cache-name product_memory_bank_real_dataset --output-dir data/outputs/threshold_optimization_results
```

This command writes:

```text
data/outputs/threshold_optimization_results/thresholds.csv
data/outputs/threshold_optimization_results/thresholds_w_metrics.csv
```

Use `thresholds.csv` for deployment. Keep `thresholds_w_metrics.csv` for review,
debugging, and handover diagnostics.

Class-specific thresholds are unknown-aware. Unknown calibration examples are
grouped by the known `best_label` that attracts them, matching the threshold
lookup used at inference. Classes with a concentrated unknown false-accept rate
can rise above the global threshold while preserving correct-known acceptance.
Classes without excess unknown risk can still lower their threshold to recover
known examples. By default, lowering may not add any unknown false accepts, and
the optimizer limits the aggregate unknown false-accept-rate increase to `0.005`
before rolling unsafe lowered thresholds back to the global value.

Raising targets a per-class unknown false-accept rate of `0.01`, tests up to 15
upward steps, and allows at most a `0.03` drop from the class's global-threshold
correct-known acceptance rate.

These guardrails can be configured with:

```text
--dynamic-min-class-unknown-samples
--dynamic-max-added-unknown-accepts
--dynamic-max-unknown-false-accept-rate-increase
--dynamic-max-raise-steps
--dynamic-target-class-unknown-false-accept-rate
--dynamic-max-known-correct-accept-rate-drop
```

### 3. Run Inference for One Image

```powershell
python -m product_recognition.cli.run_inference --cache-dir data/outputs/memory_bank_cache --cache-name product_memory_bank_real_dataset --threshold-dir data/outputs/threshold_optimization_results --image-path data/input_images/image_1/crop_0001_conf_0.98.jpg
```

Single-image mode prints a detailed JSON result to the console.

To also write CSV files:

```powershell
python -m product_recognition.cli.run_inference --cache-dir data/outputs/memory_bank_cache --cache-name product_memory_bank_real_dataset --threshold-dir data/outputs/threshold_optimization_results --image-path data/input_images/image_1/crop_0001_conf_0.98.jpg --output-csv data/outputs/inference_results/inference_results.csv
```

### 4. Run Inference for a Folder

```powershell
python -m product_recognition.cli.run_inference --cache-dir data/outputs/memory_bank_cache --cache-name product_memory_bank_real_dataset --threshold-dir data/outputs/threshold_optimization_results --image-dir data/input_images --output-csv data/outputs/inference_results/inference_results.csv
```

When `--output-csv` is provided, folder inference writes two files:

```text
data/outputs/inference_results/inference_results.csv
data/outputs/inference_results/inference_results_detailed.csv
```

`inference_results.csv` is the compact client-facing output:

```text
crop_path,final_pred
```

`inference_results_detailed.csv` contains the full debug output, including scores,
thresholds, ranked classes, neighbors, and rejection details.

## Python Backend Integration

For a Python backend, load the recognizer once at service startup and reuse it
for all requests:

```python
from product_recognition.inference import load_recognizer, predict_image

recognizer = load_recognizer(
    cache_dir="data/outputs/memory_bank_cache",
    cache_name="product_memory_bank_real_dataset",
    threshold_dir="data/outputs/threshold_optimization_results",
)

result = predict_image(recognizer, "path/to/crop.jpg")

print(result["final_pred"])
print(result["is_rejected"])
print(result["best_score"])
```

Use `is_rejected` as the authoritative flag for unknown products. If
`is_rejected` is `True`, `final_pred` will be `__Unknown__`, while `best_label`
still shows the closest known product for debugging.

## Important Output Fields

The detailed Python/API result includes:

`final_pred`

Final product prediction. This is either a known `brand__product` label or
`__Unknown__`.

`is_rejected`

Boolean flag showing whether the prediction was rejected as unknown.

`reject_reasons`

List of rejection reasons. Currently the main reason is `low_score`.

`best_label`

Nearest accepted candidate before threshold rejection. Useful for debugging.

`best_score`

Aggregated nearest-neighbor score for `best_label`.

`effective_score_threshold`

Threshold applied to the best label. This can be the global threshold or a
class-specific threshold.

`threshold_offset`

Difference between the effective class threshold and the global threshold.

`second_label`, `second_score`, `margin`

Second-best class information and score gap. Useful for reviewing ambiguous
predictions.

`ranked_classes`, `neighbors`

Detailed retrieval/debug information. These fields are intentionally excluded
from the compact inference CSV.

## Updating Artifacts

If the product dataset changes, rebuild the artifacts in this order:

1. Run `create_memory_bank`.
2. Run `optimize_thresholds`.
3. Run `run_inference` on validation or sample images.
4. Replace the deployed memory-bank cache and threshold folder together.

Do not mix thresholds from one cache with a different memory-bank cache unless
they were intentionally generated from the same dataset/version.

## Runtime Notes

- Default model: `Qwen/Qwen3-VL-Embedding-2B`.
- Default unknown label: `__Unknown__`.
- Default nearest-neighbor count: `top_k=5`.
- Embeddings are L2-normalized before storage and search.
- FAISS uses inner product, equivalent to cosine similarity for normalized vectors.
- The memory-bank cache is a Python pickle file. Load only trusted internal artifacts.
- For production use, load the recognizer once and reuse it instead of reloading per image.
