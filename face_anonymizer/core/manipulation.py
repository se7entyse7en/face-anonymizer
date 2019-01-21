from typing import Tuple

import cv2
import numpy as np

from face_anonymizer.core.detection import BoundingBox


Color = Tuple[int, int, int]


class Drawing:
    """Utility to draw on images"""

    @staticmethod
    def bounding_boxes(image: np.ndarray, *bboxes: BoundingBox,
                       color: Color = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """Draw the bounding boxes on the given image

        Args:
            image: the image we want to draw on
            bboxes: the bounding boxes to draw
            color: the color to use for the boxes and the text
            thickness: the thickness of the boxes and the text
        Returns:
            the modified image

        """
        image = image.copy()
        for bbox in bboxes:
            text = '{:.2f}%'.format(bbox.confidence * 100)
            start_x, start_y = bbox.top_left
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(image, text, (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)
            cv2.rectangle(image, tuple(bbox.top_left),
                          tuple(bbox.bottom_right),
                          color, thickness)
        return image

    @staticmethod
    def points(image: np.ndarray, *points: Tuple[int, int],
               color: Color = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw the given points on the given image

        Args:
            image: the image we want to draw on
            points: the points to draw as (x, y) coordinates
            color: the color to use for the points
            thickness: the thickness of the points
        Returns:
            the modified image

        """
        image = image.copy()
        for x, y in points:
            cv2.circle(image, (x, y), 2, (255, 0, 0), cv2.FILLED)

        return image


class Manipulator:
    """Utility to manipulate images"""

    @staticmethod
    def pixelate(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Pixelate the given image with the given strength

        Args:
            pixelate: the image we want to pixelate
            points: the points to draw as (x, y) coordinates
        Returns:
            the modified image

        """
        downsampled = cv2.resize(image, None, fx=1/strength, fy=1/strength,
                                 interpolation=cv2.INTER_AREA)
        return cv2.resize(downsampled, image.shape[:2][::-1],
                          interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def replace_bounding_box(image: np.ndarray, bbox: BoundingBox,
                             replace: np.ndarray) -> None:
        """Replace the region the bounding box with a same-sized region

        Args:
            pixelate: the image we want to pixelate
            points: the points to draw as (x, y) coordinates

        """
        if bbox.size != replace.shape[:2][::-1]:
            raise ValueError('The bounding box to replace and the replacement '
                             'have different sizes')

        image[bbox.top_left[1]:bbox.bottom_left[1],
              bbox.top_left[0]:bbox.top_right[0]] = replace
