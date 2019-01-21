from typing import Optional

import cv2
import numpy as np

from face_anonymizer.core.detection import FaceDetector
from face_anonymizer.core.detection import FaceExtractor
from face_anonymizer.core.manipulation import Manipulator


def pixelate_faces_from_path(fd: FaceDetector, image_path: str,
                             detection_threshold: Optional[float] = None,
                             strength: int = 10) -> np.ndarray:
    """Pixelates the faces in the image at the provided path

    Args:
        fd: an instance of `FaceDetector`
        image_path: the path of the image we want to pixelate the faces
        detection_threshold: minimum confidence to consider a region as a face
        strength: pixelation strength
    Returns:
        the modified image

    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return pixelate_faces(fd, image, detection_threshold=detection_threshold,
                          strength=strength)


def pixelate_faces(fd: FaceDetector, image: np.ndarray,
                   detection_threshold: Optional[float] = None,
                   strength: int = 10, copy: bool = False) -> np.ndarray:
    """Pixelates the faces in the provided image

    Args:
        fd: an instance of `FaceDetector`
        image: the image we want to pixelate the faces
        detection_threshold: minimum confidence to consider a region as a face
        strength: pixelation strength
        copy: whether to modify a copy of the provided image
    Returns:
        the modified image which is a copy if `copy` is `True`

    """
    bboxes = list(fd.detect(image, threshold=detection_threshold))
    faces = FaceExtractor.extract(image, *bboxes)
    faces_pixelated = (Manipulator.pixelate(face) for face in faces)

    if copy:
        image = image.copy()

    for bbox, fp in zip(bboxes, faces_pixelated):
        Manipulator.replace_bounding_box(image, bbox, fp)

    return image
