"""
vessel_infer.py

Faithful reproduction of the vessel (R2-V2 / RRWNet) inference pipeline as
ACTUALLY EXECUTED in vessels.ipynb (the "FAST CPU TEST" path in cells 6 and
9 -- NOT the reflect-padding `infer.py` reference code that was only
printed/browsed, never run on real data).

Pipeline (see PHASE 2 comparison table for full [K]/[I]/[A] provenance):

    RGB image (skimage.io.imread) /255
      -> resize width to 1408 (skimage.transform.resize, anti_aliasing=True)
      -> FOV mask = (cfp.sum(axis=2) > 0.01)
      -> r2v2_preprocessing.preprocess_img  (enhance_image + CLAHE)
      -> concat [preprocessed(3ch), raw_cfp(3ch)] = 6 channels
      -> resize whole 6ch tensor to (480, 720)      <-- APPROXIMATION, see below
      -> ONNX inference -> (1,3,480,720) raw logits
      -> sigmoid
      -> * FOV mask (resized to 480x720, nearest)
      -> take channel 2 (blood-vessel union)
      -> resize prediction back to cfp (1408-width) resolution, INTER_LINEAR
      -> normalize by per-image max (/max()+1e-8)
      -> 20px FOV-boundary erosion suppression (cfp-resolution FOV)
      -> resize to ORIGINAL resolution, INTER_LINEAR
      -> threshold >= 0.35 -> binary mask

APPROXIMATION NOTE (explicitly requested to proceed with):
The notebook resizes the 6-channel tensor to (480, 720) using
`torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`.
The Mac environment has no PyTorch, so this one resize step is done with
`cv2.resize(..., interpolation=cv2.INTER_LINEAR)` instead. This is a close
but NOT bit-identical approximation of PyTorch bilinear with
align_corners=False. No other resize in this file is a substitution --
every other resize already used skimage/cv2 in the notebook itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import io as skio
from skimage.transform import resize as sk_resize

import r2v2_preprocessing
from common import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PipelineError,
    check_output_shape,
    describe_session,
    load_onnx_session,
    require_file,
    save_mask_png,
    sigmoid,
)

VESSEL_ONNX = MODELS_DIR / "Vessel_model.onnx"
NETWORK_HW = (480, 720)          # (H, W) expected by the ONNX graph
VESSEL_THRESHOLD = 0.35
BOUNDARY_BAND_PX = 20
TARGET_WIDTH = 1408


def run_vessel_inference(image_path: Path, session=None) -> np.ndarray:
    """Returns the final vessel binary mask (uint8, 0/255) at ORIGINAL
    image resolution."""

    require_file(image_path, "Vessel input image")

    if session is None:
        session = load_onnx_session(VESSEL_ONNX, needs_external_data=True)
        describe_session(session, "Vessel")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ------------------------------------------------------------
    # 1. Load image, scale to 0-1
    # ------------------------------------------------------------
    original = skio.imread(str(image_path)).astype(np.float32) / 255.0
    original = original[:, :, :3]
    original_h, original_w = original.shape[:2]

    # ------------------------------------------------------------
    # 2. Resize width to 1408 (skimage, anti-aliased) -- matches notebook
    # ------------------------------------------------------------
    cfp = original.copy()
    if cfp.shape[1] != TARGET_WIDTH:
        new_h = int(cfp.shape[0] * TARGET_WIDTH / cfp.shape[1])
        cfp = sk_resize(
            cfp, (new_h, TARGET_WIDTH, 3),
            preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

    # ------------------------------------------------------------
    # 3. FOV mask + R2-V2 preprocessing (enhance_image + CLAHE)
    # ------------------------------------------------------------
    fov_seed = (cfp.sum(axis=2) > 0.01).astype(np.float32)
    pre, mask_cfp = r2v2_preprocessing.preprocess_img(cfp, fov_seed)
    # pre: float (0-1), CLAHE-equalized, shape (h,1408,3)
    # mask_cfp: uint8 (0/255), eroded FOV mask at cfp resolution

    # ------------------------------------------------------------
    # 4. Build 6-channel input: [preprocessed(3ch), raw_cfp(3ch)]
    # ------------------------------------------------------------
    img6 = np.concatenate([pre.astype(np.float32), cfp.astype(np.float32)], axis=-1)

    # ------------------------------------------------------------
    # 5. Resize whole 6ch tensor to network resolution (480, 720)
    #    APPROXIMATION: cv2 INTER_LINEAR in place of torch bilinear
    #    align_corners=False (see module docstring).
    # ------------------------------------------------------------
    net_h, net_w = NETWORK_HW
    img6_resized = np.stack(
        [cv2.resize(img6[:, :, c], (net_w, net_h), interpolation=cv2.INTER_LINEAR)
         for c in range(6)],
        axis=-1,
    ).astype(np.float32)

    mask_net = cv2.resize(mask_cfp, (net_w, net_h), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------
    # 6. ONNX inference
    # ------------------------------------------------------------
    input_tensor = img6_resized.transpose(2, 0, 1)[None].astype(np.float32)
    check_output_shape(input_tensor.shape, (1, 6, net_h, net_w), "Vessel ONNX input")

    raw = session.run([output_name], {input_name: input_tensor})[0]
    check_output_shape(raw.shape, (1, 3, net_h, net_w), "Vessel ONNX output")

    probs = sigmoid(raw)
    probs = probs * (mask_net > 0).astype(np.float32)[None, None, :, :]

    vessel_map = probs[0, 2]  # channel 2 = blood-vessel union

    # ------------------------------------------------------------
    # 7. Resize prediction back to cfp (1408-width) resolution
    # ------------------------------------------------------------
    vessel_map_cfp = cv2.resize(
        vessel_map, (cfp.shape[1], cfp.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    # ------------------------------------------------------------
    # 8. Per-image max normalization
    # ------------------------------------------------------------
    vessel_norm = vessel_map_cfp / (vessel_map_cfp.max() + 1e-8)

    # ------------------------------------------------------------
    # 9. FOV-boundary suppression (20px erosion band), at cfp resolution
    # ------------------------------------------------------------
    fov = (mask_cfp > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * BOUNDARY_BAND_PX + 1, 2 * BOUNDARY_BAND_PX + 1),
    )
    inner_fov = cv2.erode(fov, kernel, iterations=1)
    boundary_band = (fov > 0) & (inner_fov == 0)

    clean_vessel_map = vessel_norm.copy()
    clean_vessel_map[boundary_band] = 0
    clean_vessel_map[fov == 0] = 0

    # ------------------------------------------------------------
    # 10. Resize to ORIGINAL resolution
    # ------------------------------------------------------------
    clean_original = cv2.resize(
        clean_vessel_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR
    )

    # ------------------------------------------------------------
    # 11. Threshold
    # ------------------------------------------------------------
    binary_mask = (clean_original >= VESSEL_THRESHOLD).astype(np.uint8) * 255

    return binary_mask, clean_original


def main():
    parser = argparse.ArgumentParser(description="Vessel (R2-V2) ONNX inference")
    parser.add_argument("image_path", type=str, help="Path to input fundus image")
    parser.add_argument("--out", type=str, default=None,
                         help="Output mask path (default: dr_onnx/outputs/Vessel_mask.png)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "Vessel_mask.png"

    try:
        binary_mask, prob_map = run_vessel_inference(image_path)
    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}")
        raise SystemExit(1)

    save_mask_png(binary_mask, out_path)
    np.save(str(out_path.with_name("Vessel_probability.npy")), prob_map)
    print("Vessel inference complete.")


if __name__ == "__main__":
    main()
