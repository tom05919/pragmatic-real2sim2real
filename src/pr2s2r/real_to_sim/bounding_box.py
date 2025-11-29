"""
Robust bounding box detection using Color Thresholding.

Strategy:
1. Convert image to HSV color space.
2. Threshold specifically for the gold/yellow/brown color of the boxes.
   This ignores white paper, gray shadows, and other background noise.
3. Use morphological closing to merge the two boxes into one L-shape.
4. Measure dimensions.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectMeasurement:
    """Stores measurement results."""
    shape_type: str
    longest_width: float
    longest_height: float
    box_corners: np.ndarray


def get_object_mask(img: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for the object using HSV color thresholding.
    Targeting the yellow/gold/brown color of the Pocky boxes.
    """
    # 1. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Define color ranges for the "Gold/Yellow/Brown" boxes
    # Hue is 0-180 in OpenCV. Yellow is around 30.
    # We want to catch:
    #   - Yellows (Hue ~20-40)
    #   - Browns/Oranges (Hue ~10-25)
    #   - Saturation must be moderate to high (ignore white/gray paper)
    #   - Value/Brightness can vary
    
    # Range 1: Yellow/Orange/Gold
    lower_gold = np.array([10, 50, 50])   # Hue 10-40, S>50 (ignore white), V>50 (ignore black)
    upper_gold = np.array([40, 255, 255])
    
    # Range 2: Darker Brown (Chocolate parts)
    # Brown is basically dark orange
    lower_brown = np.array([0, 40, 40]) 
    upper_brown = np.array([20, 255, 200])
    
    mask1 = cv2.inRange(hsv, lower_gold, upper_gold)
    mask2 = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 3. Clean up the mask
    # Remove small noise (speckles)
    kernel_small = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # 4. Merge nearby objects
    # Use a very large kernel to bridge the gap between the two boxes
    # and fill in any holes (like the text on the box)
    kernel_large = np.ones((35, 35), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    
    # 5. Fill holes (optional but good for solid objects)
    # Find contours and fill the largest one to make it solid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Fill the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
    return mask


def measure_object(
    image_path: str, output_path: str = "output.png", debug: bool = False
) -> ObjectMeasurement:
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Work on a copy for visualization
    vis_img = img.copy()

    # 2. Get Object Mask
    mask = get_object_mask(img)

    # 3. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If we found nothing, save the debug mask to see why
        cv2.imwrite("debug_mask_failed.png", mask)
        raise ValueError("No object detected. Check 'debug_mask_failed.png'.")
        
    # Pick the largest contour (the main object)
    contour = max(contours, key=cv2.contourArea)
    
    # 4. Measure
    # Compute minimum area rectangle (rotated bounding box)
    rect = cv2.minAreaRect(contour)
    (center, (w, h), angle) = rect
    
    # Get corners of the box
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    longest_width = max(w, h)
    longest_height = min(w, h)
    
    # 5. Classify Shape
    hull = cv2.convexHull(contour)
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = contour_area / hull_area if hull_area > 0 else 0
    
    # L-shapes are concave (solidity < ~0.85)
    if solidity < 0.85:
        shape_type = "l_shape"
    else:
        shape_type = "rectangle"
        
    measurement = ObjectMeasurement(
        shape_type=shape_type,
        longest_width=longest_width,
        longest_height=longest_height,
        box_corners=box
    )

    # 6. Visualization
    # Draw the contour (green)
    cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
    
    # Draw the bounding box (red)
    cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 3)
    
    # Add text
    label = f"{shape_type.upper()}: {longest_width:.1f}x{longest_height:.1f}px"
    cv2.putText(vis_img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    if debug:
        cv2.imwrite("debug_mask.png", mask)
        
    cv2.imwrite(output_path, vis_img)
    
    return measurement


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.jpeg")
    output_path = os.path.join(script_dir, "boxed_output.png")

    try:
        m = measure_object(input_path, output_path, debug=True)
        print(f"Detected: {m.shape_type}")
        print(f"Dimensions: {m.longest_width:.1f} x {m.longest_height:.1f}")
        print(f"Output saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
