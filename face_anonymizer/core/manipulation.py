from typing import Tuple

import cv2
import numpy as np

from face_anonymizer.core.detection import BoundingBox


Color = Tuple[int, int, int]


class Drawing:

    @staticmethod
    def bounding_boxes(image: np.ndarray, *bboxes: BoundingBox,
                       color: Color = (0, 255, 0), thickness: int = 2) -> None:
        for bbox in bboxes:
            text = "{:.2f}%".format(bbox.confidence * 100)
            start_x, start_y = bbox.top_left
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(image, text, (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)
            cv2.rectangle(image, tuple(bbox.top_left),
                          tuple(bbox.bottom_right),
                          color, thickness)

    @staticmethod
    def points(image: np.ndarray, *points: Tuple[int, int],
               color: Color = (0, 255, 0), thickness: int = 2) -> None:
        for x, y in points:
            cv2.circle(image, (x, y), 2, (255, 0, 0), cv2.FILLED)


class Manipulator:

    @staticmethod
    def pixelate(image: np.ndarray, strength: int = 10) -> np.ndarray:
        downsampled = cv2.resize(image, None, fx=1/strength, fy=1/strength,
                                 interpolation=cv2.INTER_AREA)
        return cv2.resize(downsampled, image.shape[:2][::-1],
                          interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def replace_bounding_box(image: np.ndarray, bbox: BoundingBox,
                             replace: np.ndarray) -> None:
        if bbox.size != replace.shape[:2][::-1]:
            raise ValueError('xxx')

        image[bbox.top_left[1]:bbox.bottom_left[1],
              bbox.top_left[0]:bbox.top_right[0]] = replace
