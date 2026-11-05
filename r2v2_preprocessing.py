"""
r2v2_preprocessing.py

FAITHFUL reproduction of the R2-V2 `preprocessing.py` module exactly as
printed from within vessels.ipynb (cell that does
`print(open("/content/r2v2/preprocessing.py").read())`).

This is copied close to verbatim, with only the CLI __main__ block removed
(it is not used by our inference pipeline). Everything that affects
numerical output is untouched.

IMPORTANT / DO NOT "FIX":
`enhance_image()` divides `img` by 255 even when the caller (vessel_infer.py)
already passes an image scaled to 0-1 (via `io.imread(...).astype(np.float32) / 255.0`).
This means the image is effectively divided by 255 twice before the zoom /
erosion / Gaussian-subtraction steps. This is a genuine quirk of the working
notebook, confirmed by reading the actual preprocessing.py source. Per the
project rules, it is reproduced as-is and NOT corrected.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import skimage.io as io
from skimage.exposure import equalize_adapthist
from scipy.interpolate import interp1d
from skimage.morphology import disk


def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def to_0_1(img):
    interp_fun = interp1d([img.min(), img.max()], [0.0, 1.0])
    return interp_fun(img)


def enhance_image(img, mask, int_format=False, disk_size=5):
    """Enhance an image using the method described in the R2-V2 paper.

    Args:
        img (uint8 np.ndarray or 0-1 float np.ndarray): Image to enhance.
            NOTE: this function internally divides by 255 regardless of the
            input's existing scale (see module docstring above).
        mask (np.ndarray): ROI mask of the image to enhance.

    Returns:
        Enhanced image, enhanced mask.
    """
    # Read image and its corresponding mask
    if isinstance(img, str) or isinstance(img, Path):
        img = io.imread(img)[..., :3]
    if isinstance(mask, str) or isinstance(mask, Path):
        mask = io.imread(mask)

    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    # mask = np.where(mask > (255//2), 255, 0)
    mask = np.where(mask > 0.5, 1, 0)

    # Copy original image
    img_copy = img.copy()
    # Convert to PIL format
    zoomed_image = Image.fromarray(np.uint8(img_copy * 255))
    # Enlarge image
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1] * 1.15), int(img_copy.shape[0] * 1.15)),
        Image.BICUBIC  # type: ignore
    )
    # To numpy array type
    zoomed_image = np.array(zoomed_image)
    # Crop image to original size (zoom result)
    zoomed_image = crop_center(zoomed_image, img_copy.shape[1],
                                img_copy.shape[0])
    # Convert image from 0-255 format to 0.0-1.0 format
    zoomed_image = zoomed_image / 255.0

    # Create circular kernel for mask erosion
    kernel = disk(disk_size)

    # Erode mask
    mask = ndimage.binary_erosion(mask, kernel)
    # Convert boolean array to float array
    mask = mask * 1.0  # type: ignore

    img_copy[mask < 1.0] = 0.0

    # Create RGB mask (same mask for all channels)
    mask = np.stack((mask, mask, mask), axis=2)

    composed_image = mask.copy()
    composed_image[mask == 1.0] = img_copy[mask == 1.0]
    composed_image[mask < 1.0] = zoomed_image[mask < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask < 1.] = 0.

    enhanced_image = subtracted_image / np.std(subtracted_image)
    enhanced_image = to_0_1(enhanced_image)
    enhanced_image[mask < 1.] = 0.

    mask = mask[:, :, 0]

    if int_format:
        enhanced_image *= 255
        enhanced_image = enhanced_image.astype(np.uint8)
        mask *= 255
        mask = mask.astype(np.uint8)

    return enhanced_image, mask


def preprocess_img(img, mask):
    img_enh, mask_enh = enhance_image(img, mask, int_format=True)
    img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
    return img_enh_clahe, mask_enh
