import cv2
import numpy as np

def find_pupil_center(image, min_radius=10, max_radius=60):
    """
    Use Hough Circle Transform to find the pupil center in a clean
    grayscale IR eye image.

    Args:
        image:      normalized grayscale 128x128 image
        min_radius: minimum expected pupil radius in pixels
        max_radius: maximum expected pupil radius in pixels

    Returns:
        center: (x, y) pupil center as floats, or None if not found
        radius: pupil radius in pixels, or None if not found
    """
    # Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        print("[roi] No pupil circle found — frame will be discarded")
        return None, None

    circles = np.round(circles[0, :]).astype(int)

    # If multiple circles found, pick the darkest one (pupil = darkest region)
    best_circle    = None
    darkest_mean   = 255

    for (x, y, r) in circles:
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_brightness = cv2.mean(image, mask=mask)[0]

        if mean_brightness < darkest_mean:
            darkest_mean = mean_brightness
            best_circle  = (x, y, r)

    cx, cy, radius = best_circle
    print(f"[roi] Pupil found → center=({cx}, {cy}) "
          f"radius={radius}px brightness={darkest_mean:.1f}")

    return (cx, cy), radius

def draw_detection(image, center, radius):
    """
    Draw the Hough detected circle and center point on the image for verification.

    Args:
        image:  grayscale 128x128 processed image
        center: (x, y) pupil center from find_pupil_center()
        radius: pupil radius from find_pupil_center()

    Returns:
        vis: BGR image with circle and center point drawn
    """
    # Convert grayscale to BGR so we can draw in color
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cx, cy = center

    # Draw the detected circle in green
    cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 1)

    # Draw center point in red (small filled circle)
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)

    return vis