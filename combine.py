"""
combine.py

Combines the three binary masks (vessel, optic disc, hard exudates) into a
single non-overlapping labelled mask:

    0 = background
    1 = retinal vessels
    2 = optic disc
    3 = hard exudates

Deterministic overlap priority (IMPLEMENTATION RULE ONLY, not a clinical
claim): optic disc > hard exudates > vessels.

Also writes a separate colored `combined_overlay.png` for human viewing.
The labelled `combined_mask.png` itself stores raw integer class values
(0-3) and must NOT be confused with a stretched/colorized visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import OUTPUTS_DIR, PipelineError, require_file

LABEL_BACKGROUND = 0
LABEL_VESSEL = 1
LABEL_OD = 2
LABEL_EX = 3

# Overlay colors (BGR, for cv2) -- purely for visualization
COLOR_VESSEL = (255, 255, 255)   # white
COLOR_OD = (255, 0, 0)           # blue
COLOR_EX = (0, 0, 255)           # red


def _load_binary_mask(path: Path, label: str) -> np.ndarray:
    require_file(path, f"{label} mask")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise PipelineError(f"[UNREADABLE MASK] Could not read {label} mask: {path}")
    return mask


def combine_masks(vessel_path: Path, od_path: Path, ex_path: Path,
                   original_image_path: Path | None = None):
    vessel = _load_binary_mask(vessel_path, "Vessel")
    od = _load_binary_mask(od_path, "OD")
    ex = _load_binary_mask(ex_path, "EX")

    shapes = {"vessel": vessel.shape, "od": od.shape, "ex": ex.shape}
    if not (vessel.shape == od.shape == ex.shape):
        raise PipelineError(
            f"[DIMENSION MISMATCH] All three masks must share the same "
            f"original-resolution dimensions before combining. Got: {shapes}"
        )

    h, w = vessel.shape

    combined = np.zeros((h, w), dtype=np.uint8)
    # Priority: optic disc > hard exudates > vessels
    combined[vessel > 0] = LABEL_VESSEL
    combined[ex > 0] = LABEL_EX
    combined[od > 0] = LABEL_OD

    # ------------------------------------------------------------
    # Colored overlay for human viewing (separate from the label mask)
    # ------------------------------------------------------------
    if original_image_path is not None and Path(original_image_path).exists():
        base = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
        if base is not None and base.shape[:2] != (h, w):
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)

    overlay = base.copy()
    overlay[combined == LABEL_VESSEL] = COLOR_VESSEL
    overlay[combined == LABEL_EX] = COLOR_EX
    overlay[combined == LABEL_OD] = COLOR_OD  # drawn last -> highest visual priority too

    blended = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)

    return combined, blended


def main():
    parser = argparse.ArgumentParser(description="Combine vessel/OD/EX masks into one labelled mask")
    parser.add_argument("--vessel", type=str, default=str(OUTPUTS_DIR / "Vessel_mask.png"))
    parser.add_argument("--od", type=str, default=str(OUTPUTS_DIR / "OD_mask.png"))
    parser.add_argument("--ex", type=str, default=str(OUTPUTS_DIR / "EX_mask.png"))
    parser.add_argument("--image", type=str, default=None,
                         help="Optional original fundus image, for the overlay visualization")
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "combined_mask.png"))
    parser.add_argument("--overlay-out", type=str, default=str(OUTPUTS_DIR / "combined_overlay.png"))
    args = parser.parse_args()

    try:
        combined, overlay = combine_masks(
            Path(args.vessel), Path(args.od), Path(args.ex),
            Path(args.image) if args.image else None,
        )
    except PipelineError as e:
        print(f"\nPIPELINE STOPPED: {e}")
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)  # raw 0-3 label values, single channel
    print(f"Saved combined label mask: {out_path}  "
          f"unique values={np.unique(combined)} (0=bg,1=vessel,2=OD,3=EX)")

    overlay_path = Path(args.overlay_out)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Saved colored overlay: {overlay_path}")


if __name__ == "__main__":
    main()
