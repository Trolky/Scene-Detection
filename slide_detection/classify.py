"""Segment reclassification — identifying 'camera' and 'demo' among 'slide' segments."""
import logging
from typing import Dict, List

import cv2
import numpy as np

from .config import DetectorConfig
from .imaging import format_time
from .video import VideoSource

logger = logging.getLogger("SlideDetector")


def _has_demo_overlay(frame_bgr: np.ndarray) -> bool:
    """
    Returns True when a frame appears to show an application window (terminal/IDE)
    overlaid on a light presentation slide.

    Heuristic: the frame must contain both a significant very-dark area (the
    overlay, brightness < 60) and a significant very-light area (the slide
    background, brightness > 200). Additionally the dark pixels must be
    spatially concentrated (high standard deviation across grid cells) rather
    than spread evenly like normal text on a slide.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    total = gray.size
    dark_ratio = np.count_nonzero(gray < 60) / total
    light_ratio = np.count_nonzero(gray > 200) / total

    if dark_ratio < 0.04 or light_ratio < 0.25:
        return False

    grid = 4
    h, w = gray.shape
    cell_h, cell_w = h // grid, w // grid
    dark_block_ratios = np.array([
        np.count_nonzero(gray[r * cell_h:(r + 1) * cell_h,
                              c * cell_w:(c + 1) * cell_w] < 60) / (cell_h * cell_w)
        for r in range(grid) for c in range(grid)
    ], dtype=float)
    return float(np.std(dark_block_ratios)) > 0.12


class SlideClassifier:
    """Reclassifies 'slide' segments as 'camera' or 'demo' based on sampled frame content.

    Holds loaded Haar cascades (only when face detection is enabled — lazy load).
    """

    def __init__(self, video: VideoSource, config: DetectorConfig):
        self.video = video
        self.config = config
        self._face_cascade = None
        self._profile_cascade = None
        if config.use_face_detection:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )

    def reclassify_by_face(self, slides: List[Dict]) -> List[Dict]:
        """
        Post-processing pass that reclassifies 'slide' segments as 'camera' when a face
        is detected in sampled frames.

        Samples 3 frames spread across the stable portion of each slide segment and runs
        Haar Cascade face detection on each. If any sample contains a face whose area
        exceeds face_area_threshold (relative to frame area), the segment is reclassified
        as 'camera'. This catches stable full-screen camera shots that don't generate
        many rapid transitions and are therefore missed by merge_camera_segments.

        Skips segments already classified as 'camera'. Runs in O(n_slides) time since
        Haar Cascade takes ~5 ms per frame on CPU.
        """
        cascades = [c for c in (self._face_cascade, self._profile_cascade)
                    if c is not None and not c.empty()]

        logger.info("\n[Post-processing: camera reclassification (face + color)...]")
        for slide in slides:
            if slide["type"] == "camera":
                continue

            content_start = slide.get("content_start", slide["start"])
            stable_duration = slide["end"] - content_start
            margin = min(0.5, stable_duration * 0.15)
            t_start = content_start + margin
            t_end = slide["end"] - margin

            cap = cv2.VideoCapture(self.video.path)
            reason = None
            for k in range(3):
                t = t_start + (t_end - t_start) * k / 2 if t_end > t_start else t_start
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

                # --- color saturation heuristic ---
                # Fullscreen camera has skin tones + natural background across most of
                # the frame. PiP camera in a corner only raises saturation in a small
                # region (~15-20 % of frame), so the pixel ratio stays well below 0.40.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                sat = hsv[:, :, 1]
                if float(np.count_nonzero(sat > 60) / sat.size) > 0.40:
                    reason = "color saturation"
                    break

                # --- Haar Cascade face detection (frontal + profile) ---
                if cascades:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_eq = cv2.equalizeHist(gray)
                    h, w = frame.shape[:2]
                    frame_area = h * w
                    for cascade in cascades:
                        for img in (gray_eq, gray):
                            faces = cascade.detectMultiScale(
                                img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                            )
                            for (_, _, fw, fh) in faces:
                                if (fw * fh) / frame_area >= self.config.face_area_threshold:
                                    reason = "face detected"
                                    break
                            if reason:
                                break
                        if reason:
                            break
                if reason:
                    break
            cap.release()

            if reason:
                logger.info(f"  Slide {slide['id']} [{format_time(slide['start'])} – "
                      f"{format_time(slide['end'])}] → {reason} → camera")
                slide["type"] = "camera"

        return slides

    def reclassify_demo_slides(self, slides: List[Dict]) -> List[Dict]:
        """
        Reclassifies 'slide' segments as 'demo' when the representative frame
        contains a dark application-window overlay on a light slide background.
        Runs after face-detection reclassification so camera segments are skipped.
        """
        logger.info("\n[Post-processing: demo-overlay detection...]")
        for slide in slides:
            if slide["type"] != "slide":
                continue
            content_start = slide.get("content_start", slide["start"])
            mid_time = (content_start + slide["end"]) / 2
            frame = self.video.read_frame_at(mid_time)
            if frame is not None and _has_demo_overlay(frame):
                logger.info(f"  Slide {slide['id']} [{format_time(slide['start'])} – "
                      f"{format_time(slide['end'])}] → demo overlay → reclassified as demo")
                slide["type"] = "demo"
        return slides