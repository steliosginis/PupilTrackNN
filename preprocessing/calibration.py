import cv2
import numpy as np

def correct_perspective(image, cam2_angle, reference_angle=0.0):
    """
    Correct image perspective based on camera angle relative to reference.
    
    Args:
        image:           raw frame from camera 1
        cam2_angle:      angle of camera 2 in degrees (from CSV)
        reference_angle: the baseline angle (default 0 = direct view)
    
    Returns:
        corrected image as numpy array
    """
    angle_diff = cam2_angle - reference_angle
    angle_rad = np.deg2rad(angle_diff)

    h, w = image.shape[:2]
    
    # Build perspective correction matrix based on angle difference
    # This shears the image to compensate for the angled view
    shear_factor = np.tan(angle_rad) * 0.5  # scale down effect
    
    src_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    dst_points = np.float32([
        [shear_factor * h, 0],
        [w - shear_factor * h, 0],
        [w, h],
        [0, h]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(image, M, (w, h))
    
    return corrected


def calibrate_frame(image, csv_row, reference_angle=0.0):
    """
    Full calibration for a single frame using its CSV metadata.
    
    Args:
        image:           raw frame
        csv_row:         corresponding row from camera_log.csv
        reference_angle: baseline angle to correct towards
    
    Returns:
        calibrated image
    """
    cam2_angle = float(csv_row['cam2_angle'])
    prev_x = float(csv_row['cam1_prev_x'])
    prev_y = float(csv_row['cam1_prev_y'])
    
    # Perspective correction based on cam2 angle
    corrected = correct_perspective(image, cam2_angle, reference_angle)
    
    print(f"[calibration] Frame {int(csv_row['frame_id'])}: "
          f"angle={cam2_angle:.1f}° | prev_pos=({prev_x:.1f}, {prev_y:.1f})")
    
    return corrected