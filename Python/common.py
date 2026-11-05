"""
common.py

Shared utilities for the DR fundus-image inference pipeline.

- Resolves dr_onnx/models and dr_onnx/outputs relative to this file
  (this file is expected to live at dr_onnx/python/common.py).
- Provides sanity-check helpers that FAIL LOUDLY (raise / exit) instead of
  silently continuing, per the project rules:
    * model files must exist (.onnx and .onnx.data where applicable)
    * input image must exist
    * ONNX input/output names and shapes are printed and checked
    * shape mismatches stop execution rather than being "fixed" silently
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------

# dr_onnx/python/common.py -> parent.parent = dr_onnx/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PipelineError(RuntimeError):
    """Raised for any condition that should stop the pipeline instead of
    being silently patched over (missing files, shape mismatches, etc.)."""


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------

def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise PipelineError(f"[MISSING FILE] {description}: {path}")
    return path


def require_onnx_model(onnx_path: Path, needs_external_data: bool) -> Path:
    """Check an .onnx file exists, and if it references external data,
    check the matching .onnx.data file also exists. Stops (raises) rather
    than guessing or proceeding with a partial model."""
    require_file(onnx_path, "ONNX model")
    data_path = Path(str(onnx_path) + ".data")
    if needs_external_data and not data_path.exists():
        raise PipelineError(
            f"[MISSING EXTERNAL DATA] {onnx_path} references external data "
            f"but {data_path} was not found. The .onnx and .onnx.data files "
            f"must stay together in the same folder."
        )
    return onnx_path


def load_onnx_session(onnx_path: Path, needs_external_data: bool = True) -> "ort.InferenceSession":
    if ort is None:
        raise PipelineError(
            "onnxruntime is not installed. Install it with:\n"
            "  pip install onnxruntime --break-system-packages"
        )
    require_onnx_model(onnx_path, needs_external_data=needs_external_data)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session


def describe_session(session: "ort.InferenceSession", label: str) -> None:
    print(f"[{label}] ONNX inputs:")
    for inp in session.get_inputs():
        print(f"    name={inp.name!r} shape={inp.shape} dtype={inp.type}")
    print(f"[{label}] ONNX outputs:")
    for out in session.get_outputs():
        print(f"    name={out.name!r} shape={out.shape} dtype={out.type}")


def check_output_shape(actual_shape, expected_shape, label: str) -> None:
    """Stops execution (raises) if shapes differ, per project rule:
    'If an input/output shape differs from the expected shape: stop and
    report it.' No silent reshaping/broadcasting."""
    if tuple(actual_shape) != tuple(expected_shape):
        raise PipelineError(
            f"[SHAPE MISMATCH] {label}: expected {expected_shape}, got {tuple(actual_shape)}. "
            f"Stopping rather than guessing how to reconcile this."
        )


# --------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 HxWx3, using OpenCV (BGR->RGB conversion)."""
    require_file(path, "Input image")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PipelineError(f"[UNREADABLE IMAGE] cv2.imread failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a single-channel mask as a PNG. Caller controls the value range
    (e.g. 0/255 for a visual binary mask, or 0..3 for the combined label
    mask). This function does NOT rescale/colorize -- that is a deliberate
    choice so a labelled mask is never accidentally saved as a stretched
    visualization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise PipelineError(f"[WRITE FAILED] Could not write mask to: {path}")
    print(f"Saved: {path}  shape={mask.shape} dtype={mask.dtype} "
          f"unique={np.unique(mask)[:10]}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax_channel_first(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax along `axis` (default: channel axis for
    NCHW tensors)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)
