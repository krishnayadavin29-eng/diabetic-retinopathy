"""
ex_infer.py

Faithful reproduction of `predict_ex_full_image` from ex.ipynb cell 31 --
the function that actually generated the final saved EX masks in
EX_Results/masks (cell 33), NOT the similar-but-different `predict_full_image`
from cell 27.

Pipeline (NO whole-image resize -- sliding window at ORIGINAL resolution):

    RGB image, original resolution
      -> slide 512x512 window, stride 256, over (h, w)
         y_positions/x_positions built exactly as in cell 31:
           range(0, max(h-512,0)+1, 256), with a final position appended
           to guarantee the bottom/right edge is covered if not already hit
      -> each patch: pad with zeros to 512x512 if it runs past the edge
      -> /255 + ImageNet normalize
      -> ONNX inference -> (1,1,512,512) raw logits
      -> sigmoid
      -> accumulate probability (only the valid, unpadded region) + count
      -> average overlapping predictions (probability_sum / count_map)
      -> threshold >= 0.5
      -> mask stays at ORIGINAL resolution throughout -- no restoration resize

Per the notebook, MIN_COMPONENT_AREA=20 connected-component filtering exists
ONLY for building an evidence CSV in the notebook and is explicitly NOT
applied here to the saved prediction mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Python.common import (
    MODELS_DIR,
    OUTPUTS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PipelineError,
    check_output_shape,
    describe_session,
    load_image_rgb,
    load_onnx_session,
    save_mask_png,
    sigmoid,
)

EX_ONNX = MODELS_DIR / "EX_model.onnx"
PATCH_SIZE = 512
STRIDE = 256
EX_THRESHOLD = 0.5


def run_ex_inference(image_path: Path, session=None):
    """Returns (binary_mask_uint8_0_255_at_original_res, probability_map_float32)."""

    rgb = load_image_rgb(image_path)
    h, w = rgb.shape[:2]

    if session is None:
        session = load_onnx_session(EX_ONNX, needs_external_data=True)
        describe_session(session, "EX")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    probability_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # ------------------------------------------------------------
    # Patch positions -- exact logic from predict_ex_full_image (cell 31)
    # ------------------------------------------------------------
    y_positions = list(range(0, max(h - PATCH_SIZE, 0) + 1, STRIDE))
    x_positions = list(range(0, max(w - PATCH_SIZE, 0) + 1, STRIDE))

    if len(y_positions) == 0 or y_positions[-1] != h - PATCH_SIZE:
        y_positions.append(max(h - PATCH_SIZE, 0))
    if len(x_positions) == 0 or x_positions[-1] != w - PATCH_SIZE:
        x_positions.append(max(w - PATCH_SIZE, 0))

    n_patches = len(y_positions) * len(x_positions)
    print(f"[EX] image size {h}x{w} -> {len(y_positions)}x{len(x_positions)} "
          f"= {n_patches} patches (patch={PATCH_SIZE}, stride={STRIDE})")

    for y in y_positions:
        for x in x_positions:
            patch = rgb[y:min(y + PATCH_SIZE, h), x:min(x + PATCH_SIZE, w)]
            ph, pw = patch.shape[:2]

            if ph != PATCH_SIZE or pw != PATCH_SIZE:
                padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            patch_float = patch.astype(np.float32) / 255.0
            patch_float = (patch_float - IMAGENET_MEAN) / IMAGENET_STD

            input_tensor = patch_float.transpose(2, 0, 1)[None].astype(np.float32)
            check_output_shape(
                input_tensor.shape, (1, 3, PATCH_SIZE, PATCH_SIZE), "EX ONNX input"
            )

            raw = session.run([output_name], {input_name: input_tensor})[0]
            check_output_shape(
                raw.shape, (1, 1, PATCH_SIZE, PATCH_SIZE), "EX ONNX output"
            )

            prob = sigmoid(raw)[0, 0]  # (512, 512)

            probability_sum[y:min(y + PATCH_SIZE, h), x:min(x + PATCH_SIZE, w)] += (
                prob[:ph, :pw]
            )
            count_map[y:min(y + PATCH_SIZE, h), x:min(x + PATCH_SIZE, w)] += 1

    probability_map = probability_sum / np.maximum(count_map, 1e-8)
    binary_mask = (probability_map >= EX_THRESHOLD).astype(np.uint8) * 255

    return binary_mask, probability_map


def main():
    parser = argparse.ArgumentParser(description="Hard exudate (EX) sliding-window ONNX inference")
    parser.add_argument("image_path", type=str, help="Path to input fundus image")
    parser.add_argument("--out", type=str, default=None,
                         help="Output mask path (default: dr_onnx/outputs/EX_mask.png)")
    parser.add_argument("--save-prob", action="store_true",
                         help="Also save EX_probability.npy for debugging")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "EX_mask.png"

    try:
        binary_mask, probability_map = run_ex_inference(image_path)
    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}")
        raise SystemExit(1)

    save_mask_png(binary_mask, out_path)
    if args.save_prob:
        np.save(str(out_path.with_name("EX_probability.npy")), probability_map)
        print(f"Saved probability map: {out_path.with_name('EX_probability.npy')}")
    print("EX inference complete.")


if __name__ == "__main__":
    main()
