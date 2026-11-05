"""
pipeline.py

End-to-end pipeline: one fundus image in, four mask files out.

Usage:
    python3 python/pipeline.py "/path/to/fundus.jpg"

Automatically locates:
    models under:  <dr_onnx>/models/   (this file's parent.parent / "models")
    outputs under: <dr_onnx>/outputs/  (this file's parent.parent / "outputs")

Produces:
    outputs/EX_mask.png
    outputs/OD_mask.png
    outputs/Vessel_mask.png
    outputs/combined_mask.png       (0=bg, 1=vessel, 2=OD, 3=EX)
    outputs/combined_overlay.png    (colored visualization)

Any missing model file, missing external .onnx.data, missing input image,
or ONNX shape mismatch STOPS the pipeline with a clear error rather than
guessing or silently continuing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import MODELS_DIR, OUTPUTS_DIR, PipelineError, describe_session, load_onnx_session
import vessel_infer
import od_infer
import ex_infer
import combine


def main():
    parser = argparse.ArgumentParser(description="Full DR fundus segmentation pipeline")
    parser.add_argument("image_path", type=str, help="Path to input fundus image")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DR fundus segmentation pipeline")
    print(f"Input image : {image_path}")
    print(f"Models dir  : {MODELS_DIR}")
    print(f"Outputs dir : {OUTPUTS_DIR}")
    print("=" * 60)

    try:
        # --------------------------------------------------------
        # Vessel
        # --------------------------------------------------------
        t0 = time.time()
        print("\n[1/4] Vessel segmentation...")
        vessel_session = load_onnx_session(vessel_infer.VESSEL_ONNX, needs_external_data=True)
        describe_session(vessel_session, "Vessel")
        vessel_mask, _ = vessel_infer.run_vessel_inference(image_path, session=vessel_session)
        vessel_out = OUTPUTS_DIR / "Vessel_mask.png"
        _save(vessel_mask, vessel_out)
        print(f"    done in {time.time()-t0:.1f}s")

        # --------------------------------------------------------
        # OD
        # --------------------------------------------------------
        t0 = time.time()
        print("\n[2/4] Optic disc segmentation...")
        od_session = load_onnx_session(od_infer.OD_ONNX, needs_external_data=True)
        describe_session(od_session, "OD")
        od_mask, _, _ = od_infer.run_od_inference(image_path, session=od_session)
        od_out = OUTPUTS_DIR / "OD_mask.png"
        _save(od_mask, od_out)
        print(f"    done in {time.time()-t0:.1f}s")

        # --------------------------------------------------------
        # EX
        # --------------------------------------------------------
        t0 = time.time()
        print("\n[3/4] Hard exudate segmentation (sliding window)...")
        ex_session = load_onnx_session(ex_infer.EX_ONNX, needs_external_data=True)
        describe_session(ex_session, "EX")
        ex_mask, _ = ex_infer.run_ex_inference(image_path, session=ex_session)
        ex_out = OUTPUTS_DIR / "EX_mask.png"
        _save(ex_mask, ex_out)
        print(f"    done in {time.time()-t0:.1f}s")

        # --------------------------------------------------------
        # Combine
        # --------------------------------------------------------
        print("\n[4/4] Combining masks...")
        combined, overlay = combine.combine_masks(vessel_out, od_out, ex_out, image_path)
        combined_out = OUTPUTS_DIR / "combined_mask.png"
        overlay_out = OUTPUTS_DIR / "combined_overlay.png"
        import cv2
        cv2.imwrite(str(combined_out), combined)
        cv2.imwrite(str(overlay_out), overlay)
        print(f"    Saved: {combined_out}")
        print(f"    Saved: {overlay_out}")

    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for f in ["Vessel_mask.png", "OD_mask.png", "EX_mask.png",
              "combined_mask.png", "combined_overlay.png"]:
        p = OUTPUTS_DIR / f
        print(f"  {p}  ({'exists' if p.exists() else 'MISSING'})")


def _save(mask, path: Path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    print(f"    Saved: {path}")


if __name__ == "__main__":
    main()
