from typing import Generator
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from dataclasses import dataclass
from dataclasses import field


class FaceExtractor:
    """Faces extractor"""

    @staticmethod
    def extract(image: np.ndarray, *bboxes: 'BoundingBox') -> Generator[
            np.ndarray, None, None]:
        """Extracts the regions corresponding to the boxes from the image

        Args:
            image: the image we want to extract the regions from
            bboxes: the bounding boxes identifying the regions to extract
        Yields:
            the regions extracted

        """
        return (
            image[box.top_left[1]:box.bottom_left[1],
                  box.top_left[0]:box.top_right[0]] for box in bboxes
        )


class FaceDetector:
    """Face detector using OpenCV DNN Face Detector"""

    # RGB mean values for ImageNet training set
    RGB_MEAN_VALUES = (104.0, 177.0, 123.0)

    def __init__(self, model_file_path: str, config_file_path: str,
                 threshold: float = 0.5):
        """Initializes the face detector

        Args:
            model_file_path: path of the model file. The original file can be
                found at the following link:

                    https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
            config_file_path: path of the config file. The original file can be
                found at the following link:

                    https://github.com/opencv/opencv/blob/3.4/samples/dnn/face_detector/opencv_face_detector.pbtxt
            threshold: minimum confidence to consider a region as a face

        """
        self._model_file_path = model_file_path
        self._config_file_path = config_file_path

        self._net = cv2.dnn.readNetFromCaffe(
            config_file_path, model_file_path)
        self._threshold = threshold

    def reinit(self):
        self._net = cv2.dnn.readNetFromCaffe(
            self._config_file_path, self._model_file_path)

    def detect_from_path(self, image_path: str,
                         threshold: Optional[float] = None) -> Tuple[
                             np.ndarray,
                             Generator['BoundingBox', None, None]
                         ]:
        """Detects faces from the image in the path provided

        The given optional threshold overrides the one set at instance level.

        Args:
            images_path: image path to detect the faces from
            threshold: minimum confidence to consider a region as a face
        Returns:
            a tuple whose first element is the image and the second
            one is a generator of `BoundingBox`es with the faces

        """
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return image, self.detect(image, threshold=threshold)

    def detect(self, image: np.ndarray,
               threshold: Optional[float] = None) -> Generator[
            'BoundingBox', None, None]:
        """Detects faces from the image

        The given optional threshold overrides the one set at instance level.

        Args:
            image: image to detect the faces from
            threshold: minimum confidence to consider a region as a face
        Yields:
            `BoundingBox`es with the faces

        """
        blob, w, h = self._blob_from_image(image)
        yield from self._detect_from_blob(blob, w, h, threshold)

    def _blob_from_image(self, image: np.ndarray) -> Tuple[
            np.ndarray, int, int]:
        h, w = image.shape[:2]

        # Depending on the performance we could resize the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), self.RGB_MEAN_VALUES,
                                     swapRB=False)

        return blob, w, h

    def _detect_from_blob(self,
                          blob: np.ndarray,
                          original_image_width: float,
                          original_image_height: float,
                          threshold: Optional[float]) -> Generator[
                              'BoundingBox', None, None]:
        threshold = threshold or self._threshold
        w, h = original_image_width, original_image_height
        self._net.setInput(blob)
        detections = self._net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')

                # Investigate why this normalization is needed
                start_x = max(0, min(start_x, w))
                end_x = max(0, min(end_x, w))

                start_y = max(0, min(start_y, h))
                end_y = max(0, min(end_y, h))

                bbox = BoundingBox.from_xy_limits(
                    start_x, end_x, start_y, end_y, confidence)

                if bbox.size[0] > 0 and bbox.size[1] > 0:
                    yield bbox


@dataclass(order=False)
class BoundingBox:
    """Bounding box identifying a rectangular region with a face"""

    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray
    confidence: Optional[int] = None
    size: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.size = (self.bottom_right[0] - self.top_left[0],
                     self.bottom_right[1] - self.top_left[1])

    def from_xy_limits(start_x: float, end_x: float,
                       start_y: float, end_y: float,
                       confidence: Optional[int] = None) -> 'BoundingBox':
        """Creates a bounding box with from the given limit values

        Args:
            start_x: left-most pixel of the region
            end_x: right-most pixel of the region
            start_y: top-most pixel of the region
            end_y: bottom-most pixel of the region
            confidence: detection confidence
        Returns:
            a `BoundingBox`

        """
        return BoundingBox(
            np.array([start_x, start_y]),
            np.array([end_x, start_y]),
            np.array([start_x, end_y]),
            np.array([end_x, end_y]),
            confidence=confidence
        )
