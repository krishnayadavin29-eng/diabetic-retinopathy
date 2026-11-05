"""
od_infer.py

Faithful reproduction of the optic-disc inference pipeline as ACTUALLY
EXECUTED in od.ipynb (cells 29/30/35/39 -- the production loops that
generated OD_Results/masks, not the earlier exploratory `autofit_resolution`
cells).

Pipeline:

    RGB image
      -> resize to 512x512, cv2.INTER_AREA
      -> /255.0
      -> ImageNet normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
      -> ONNX inference -> (1,3,512,512) raw logits
         (the exported ONNX graph is the BARE model: torch.onnx.export(od_model, ...)
          was called directly, not the `segmentation.segment()` wrapper, so
          normalization AND softmax both happen outside the graph)
      -> softmax over channel dim (manual, outside ONNX)
      -> channel 1 = optic disc probability
      -> threshold > 0.5
      -> resize mask back to ORIGINAL resolution, cv2.INTER_NEAREST

Channel 2 of the model's output is NOT identified anywhere in the notebook.
The toolkit's generic docstring ("background, CTW, EX, HE, MA") describes a
different, more general model from the same toolkit and does NOT apply to
this 3-channel OD model. Per instruction, channel 2 is treated only as
`class_2` / unlabeled -- it is never called fovea or macula anywhere in this
codebase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
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
    softmax_channel_first,
)

OD_ONNX = MODELS_DIR / "OD_model.onnx"
NET_SIZE = 512
OD_THRESHOLD = 0.5
OD_DISC_CHANNEL = 1  # confirmed [K]: channel 1 = optic disc
OD_UNLABELED_CHANNEL = 2  # NOT confirmed as fovea/macula -- "class_2" only


def run_od_inference(image_path: Path, session=None):
    """Returns (binary_mask_uint8_0_255_at_original_res, disc_prob_512, class2_prob_512)."""

    rgb = load_image_rgb(image_path)
    original_h, original_w = rgb.shape[:2]

    if session is None:
        session = load_onnx_session(OD_ONNX, needs_external_data=True)
        describe_session(session, "OD")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ------------------------------------------------------------
    # 1. Resize to 512x512, INTER_AREA
    # ------------------------------------------------------------
    img_512 = cv2.resize(rgb, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------
    # 2. /255 + ImageNet normalize
    # ------------------------------------------------------------
    img_f = img_512.astype(np.float32) / 255.0
    img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD

    # ------------------------------------------------------------
    # 3. CHW + batch dim
    # ------------------------------------------------------------
    input_tensor = img_norm.transpose(2, 0, 1)[None].astype(np.float32)
    check_output_shape(input_tensor.shape, (1, 3, NET_SIZE, NET_SIZE), "OD ONNX input")

    # ------------------------------------------------------------
    # 4. ONNX inference (raw logits, bare model)
    # ------------------------------------------------------------
    raw = session.run([output_name], {input_name: input_tensor})[0]
    check_output_shape(raw.shape, (1, 3, NET_SIZE, NET_SIZE), "OD ONNX output")

    # ------------------------------------------------------------
    # 5. Softmax over channel dim (manual)
    # ------------------------------------------------------------
    probs = softmax_channel_first(raw, axis=1)

    disc_prob = probs[0, OD_DISC_CHANNEL]          # optic disc
    class2_prob = probs[0, OD_UNLABELED_CHANNEL]    # unidentified -- NOT fovea/macula

    # ------------------------------------------------------------
    # 6. Threshold
    # ------------------------------------------------------------
    disc_binary_512 = (disc_prob > OD_THRESHOLD).astype(np.uint8)

    # ------------------------------------------------------------
    # 7. Restore to original resolution, nearest-neighbor
    # ------------------------------------------------------------
    disc_binary_full = cv2.resize(
        disc_binary_512, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    binary_mask = disc_binary_full.astype(np.uint8) * 255
    return binary_mask, disc_prob, class2_prob


def main():
    parser = argparse.ArgumentParser(description="Optic disc ONNX inference")
    parser.add_argument("image_path", type=str, help="Path to input fundus image")
    parser.add_argument("--out", type=str, default=None,
                         help="Output mask path (default: dr_onnx/outputs/OD_mask.png)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "OD_mask.png"

    try:
        binary_mask, disc_prob, class2_prob = run_od_inference(image_path)
    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}")
        raise SystemExit(1)

    save_mask_png(binary_mask, out_path)
    print("OD inference complete.")
    print("(channel 2 / 'class_2' probability map is available in-memory "
          "but not saved by default -- it is not a confirmed anatomical class.)")


if __name__ == "__main__":
    main()
