"""Video file access — metadata and frame loading."""
import logging
import os
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .imaging import preprocess_frame

logger = logging.getLogger("SlideDetector")


class VideoSource:
    """Wraps cv2.VideoCapture: holds video metadata and provides frame loading.

    Supports both sequential access (detection loop) and timestamp-based
    access (post-processing).
    """

    def __init__(self, video_path: str):
        self.path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.duration: float = 0.0

    def open(self) -> bool:
        """Opens the video and reads FPS / frame count / duration. Returns False on error."""
        if not os.path.exists(self.path):
            logger.error(f"File '{self.path}' does not exist.")
            return False

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            logger.error("Could not open video source.")
            return False

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        return True

    def release(self) -> None:
        """Releases the sequential capture. Timestamp-based loading still works after this."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def sample_step(self, check_interval: float) -> int:
        """Number of frames between two samples (always >= 1)."""
        return max(1, int(self.fps * check_interval))

    def sample_count(self, check_interval: float) -> int:
        """Estimated number of samples — used as the total count for the progress bar."""
        return self.total_frames // self.sample_step(check_interval)

    def iter_sampled_frames(self, check_interval: float) -> Iterator[Tuple[float, np.ndarray]]:
        """Yields (timestamp, raw BGR frame) approximately every check_interval seconds.

        Between samples uses cap.grab() (advances position without decoding) and only
        decodes the target frame via read() — significantly faster than fully decoding
        every frame. The first yield is the frame at t=0 (baseline), against which the
        first sample is compared.

        Args:
            check_interval: Seconds between yielded frames.

        Yields:
            Tuple of (timestamp_seconds, raw_bgr_frame).
        """
        if self._cap is None:
            return

        step = self.sample_step(check_interval)

        ret, frame = self._cap.read()
        if not ret:
            return
        yield 0.0, frame

        frame_counter = 0
        while True:
            # Skip step-1 frames using grab() — no full decode needed
            for _ in range(step - 1):
                if not self._cap.grab():
                    break
            ret, frame = self._cap.read()
            if not ret:
                break
            frame_counter += step
            yield frame_counter / self.fps, frame

    def load_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Loads and preprocesses (grayscale + blur) a single frame at a timestamp.

        Args:
            timestamp: Time in seconds.

        Returns:
            Preprocessed frame, or None on failure.
        """
        frame = self.read_frame_at(timestamp)
        return None if frame is None else preprocess_frame(frame)

    def read_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Loads a single raw BGR frame at a timestamp (no preprocessing)."""
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None