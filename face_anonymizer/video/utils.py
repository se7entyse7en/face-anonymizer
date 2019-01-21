from typing import Generator

import cv2
import numpy as np


class VideoCaptureWrapper(object):
    """A pythonic thin wrapper around `cv2.VideoCapture`"""

    def __init__(self, video_path: str):
        """Initializes the video capture wrapper

        Args:
            video_path: the path of the video

        """
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)

    @property
    def fps(self) -> float:
        """Returns the fps of the video"""

        return int(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def size(self) -> tuple:
        """Returns the size of the video"""

        return (self.width, self.height)

    @property
    def width(self) -> float:
        """Returns the width of the video"""

        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> float:
        """Returns the height of the video"""

        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def ms_interval(self) -> float:
        """Returns the real amount of ms between each frame of the video"""

        return 1000 / self.fps

    def __enter__(self) -> 'VideoCaptureWrapper':
        if not self._cap.isOpened():
            raise ValueError('Error opening video file')

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._cap.release()

    def __iter__(self) -> Generator[tuple, None, None]:
        """Iterates over the frames of the video

        Yields:
            The couples `(timer, frame)` where `timer` is the time in ms when
            the corresponding `frame` appears.

        """
        timer = 0
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                yield timer, frame
            else:
                break

            timer += self.ms_interval


class VideoWriterWrapper(object):
    """A pythonic thin wrapper around `cv2.VideoWriter`"""

    def __init__(self, video_path: str, codec: str, fps: float, size: tuple):
        """Initializes the video writer wrapper

        Args:
            video_path: the path of where to write the video
            codec: the codec of the video to write
            fps: the fps of the video to write
            size: the size of the video to write

        """
        self._video_path = video_path
        self._codec = codec
        self._fps = fps
        self._size = size

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._out = cv2.VideoWriter(video_path, fourcc, fps, size)

    def __enter__(self) -> 'VideoWriterWrapper':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._out.release()

    def write(self, frame: np.ndarray) -> None:
        """Writes the given frame

        Args:
            frame: the frame to write on the video

        """
        self._out.write(frame)
