import cv2
import numpy as np

def detect_glare(image, bright_threshold=200, max_glare_area=500):
    """
    Detect glare spots in a grayscale IR eye image.
    Handles both small specular dots and larger reflections.

    Args:
        image:            grayscale image (post-normalize)
        bright_threshold: pixel brightness above which we consider glare (0-255)
        max_glare_area:   max size of a single glare blob in pixels

    Returns:
        glare_mask:  binary image (255 = glare, 0 = clean)
        glare_count: number of distinct glare regions found
    """
    # Threshold — keep only very bright pixels
    _, bright_mask = cv2.threshold(image, bright_threshold, 255, cv2.THRESH_BINARY)

    # Dilate slightly to merge fragments of the same reflection into one blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )

    glare_mask = np.zeros_like(image)
    glare_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_glare_area:
            glare_mask[labels == i] = 255
            glare_count += 1

    # Erode back to original size to undo the dilation padding
    glare_mask = cv2.erode(glare_mask, kernel, iterations=2)

    return glare_mask, glare_count

def apply_glare_mask(image, glare_mask, method="inpaint"):
    """
    Optional: visually suppress glare for inspection purposes.
    NOT used for NN input — the raw mask is fed to the NN instead.

    Args:
        image:      grayscale image
        glare_mask: output of detect_glare()
        method:     'inpaint' fills glare with surrounding texture
                    'darken'  simply darkens glare spots to median value

    Returns:
        image with glare visually suppressed
    """
    if method == "inpaint":
        result = cv2.inpaint(image, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    elif method == "darken":
        result = image.copy()
        median_val = int(np.median(image[glare_mask == 0]))
        result[glare_mask == 255] = median_val
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def process_glare(image, bright_threshold=200, max_glare_area=150):
    """
    Main glare processing step for the pipeline.
    Returns the image unchanged + its glare mask for NN input.

    Args:
        image: normalized grayscale image

    Returns:
        image:      original image (untouched — NN learns from real data)
        glare_mask: binary mask showing glare locations
        glare_count: number of glare spots detected
    """
    glare_mask, glare_count = detect_glare(image, bright_threshold, max_glare_area)

    if glare_count > 0:
        print(f"[glare] Detected {glare_count} glare spot(s)")
    else:
        print(f"[glare] No glare detected")

    return image, glare_mask, glare_count