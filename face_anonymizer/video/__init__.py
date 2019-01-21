from typing import Optional

import cv2

from face_anonymizer.core.detection import FaceDetector
from face_anonymizer.core import pixelate_faces
from face_anonymizer.video.utils import VideoCaptureWrapper
from face_anonymizer.video.utils import VideoWriterWrapper


def pixelate_faces_from_path(fd: FaceDetector, video_path: str,
                             output_path: str, codec: Optional[str] = 'avc1',
                             detection_threshold: Optional[float] = None,
                             strength: int = 10) -> None:
    """Pixelates the faces in the video at the provided path

    Args:
        fd: an instance of `FaceDetector`
        video_path: the path of the video we want to pixelate the faces
        output_path: the output path of the video
        codec: the codec to use for writing the video
        detection_threshold: minimum confidence to consider a region as a face
        strength: pixelation strength
    Returns:
        the modified image

    """
    with VideoCaptureWrapper(video_path) as vcw:
        with VideoWriterWrapper(output_path, codec, vcw.fps, vcw.size) as vww:
            last_seconds = 0
            for timer, frame in vcw:
                current_seconds = int(timer / 1000)
                if current_seconds > last_seconds:
                    last_seconds = current_seconds
                    if last_seconds % 5 == 0:
                        print(f'Elaborated {current_seconds} seconds')

                pixelated = pixelate_faces(
                    fd, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    detection_threshold=detection_threshold,
                    strength=strength)

                vww.write(cv2.cvtColor(pixelated, cv2.COLOR_RGB2BGR))

            print('Finished')
