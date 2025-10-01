import cv2
import numpy as np
import sys
import os

BLOCK_SIZE = 20
THRESHOLD = 30
BLUR = 1    ##mora neparan broj


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply median and Gaussian blur, then invert."""
    ##image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (BLUR, BLUR), 0)
    return cv2.bitwise_not(image)


def postprocess(image: np.ndarray) -> np.ndarray:
    """Final cleanup filtering."""
    return cv2.medianBlur(image, 1)


def get_block_index(image_shape, yx, block_size):
    """Return the Y and X indices for a given block."""
    y = np.arange(max(0, yx[0] - block_size), min(image_shape[0], yx[0] + block_size))
    x = np.arange(max(0, yx[1] - block_size), min(image_shape[1], yx[1] + block_size))
    return np.meshgrid(y, x, indexing='ij')


def adaptive_median_threshold(img_in: np.ndarray) -> np.ndarray:
    """Apply median-based threshold to a block."""
    med = np.median(img_in)
    img_out = np.zeros_like(img_in, dtype=np.uint8)
    img_out[(img_in.astype(np.int16) - med) < THRESHOLD] = 255
    return img_out


def block_image_process(image: np.ndarray, block_size: int) -> np.ndarray:
    """Process image in blocks."""
    out_image = np.zeros_like(image, dtype=np.uint8)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            block_idx = get_block_index(image.shape, (row, col), block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image


def process_image_file(filename: str):
    """Main image processing pipeline."""
    image_in = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image_in is None:
        raise FileNotFoundError(f"Image not found: {filename}")

    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    ##image_out = postprocess(image_out)

    out_path = os.path.join(os.path.dirname(filename), f"bin_{os.path.basename(filename)}")
    cv2.imwrite(out_path, image_out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_file>")
        sys.exit(1)

    process_image_file(sys.argv[1])
