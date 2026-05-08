import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
from dotenv import load_dotenv
from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import (
    GeminiObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.render_wrapper import (
    RenderWrapperObjectDetector2D,
)
from prpl_perception_utils.structs import LanguageObjectDetectionID


@dataclass
class Measurement:
    total_width: float
    total_height: float
    thickness: float


def measure_object(
    image_path: str, output_path: str = "boxed_output.png"
) -> Measurement:
    load_dotenv()

    # Initialize the detector
    detector = GeminiObjectDetector2D()

    # Optionally wrap with render wrapper for visualization (examine output/ afterwards)
    detector = RenderWrapperObjectDetector2D(
        detector, outdir=Path(os.path.dirname(output_path))
    )

    # Load an image
    image = iio.imread(image_path)

    # Define objects to detect
    object_ids = [
        LanguageObjectDetectionID("horizontal pretzel box"),
        LanguageObjectDetectionID("vertical pretzel box"),
    ]

    # Run detection
    detections = detector.detect([image], object_ids)

    total_width = 0.0
    total_height = 0.0
    thickness = 0.0

    # Process results (optional printing)
    for image_detections in detections:
        for detection in image_detections:
            bbox = detection.bounding_box
            side1 = bbox.x2 - bbox.x1
            side2 = bbox.y2 - bbox.y1
            if side1 > side2:
                width = side1
                det_thickness = side2
            else:
                width = side2
                det_thickness = side1

            scaled_width = width / 1500.0
            scaled_height = det_thickness / 1500.0

            print(f"Detected {detection.object_id}")
            print(
                f"  BoundingBox: x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}"
            )
            print(f"  Dimensions: width={width}, thickness={det_thickness}")
            print(
                f"  Scaled (1/1500): width={scaled_width:.4f}, height={scaled_height:.4f}"
            )
            print(f"  with {int((detection.mask > 0).sum())} pixels in the mask")

    # Measurement logic
    if len(detections) > 0 and len(detections[0]) >= 2:
        box1 = detections[0][0].bounding_box
        box2 = detections[0][1].bounding_box
        if (
            abs(box1.y1 - box2.y1) < 10
        ):  # checks if the top of two boxes are at the same level within a 10 pixel tolerance
            total_width = (box1.x2 - box1.x1) + (box2.x2 - box2.x1)
            total_height = box2.y2 - box2.y1
            thickness = box2.x2 - box2.x1
        elif (
            abs(box1.x2 - box2.x2) < 10
        ):  # checks if the right of two boxes are at the same level within a 10 pixel tolerance
            total_width = box1.x2 - box1.x1
            total_height = (box2.y2 - box2.y1) + (box1.y2 - box1.y1)
            thickness = box1.y2 - box1.y1
    elif len(detections) > 0 and len(detections[0]) == 1:
        # Fallback for single object
        bbox = detections[0][0].bounding_box
        total_width = bbox.x2 - bbox.x1
        total_height = bbox.y2 - bbox.y1
        # thickness is not well defined for a single rect in this L-shape logic context,
        # but we can assume it's one of the dimensions or 0.
        # For a single rectangle, usually width/height are the dimensions.
        thickness = 0.0  # Or min(total_width, total_height) if needed

    print(f"Total width: {total_width}")
    print(f"Total height: {total_height}")
    print(f"Thickness: {thickness}")

    return Measurement(total_width, total_height, thickness)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.jpeg")
    output_path = os.path.join(script_dir, "boxed_output.png")
    measure_object(input_path, output_path)
