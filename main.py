import cv2
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time

class SlideDetector:
    """
    Detects presentation slides in a video file by analyzing frame differences.
    """

    def __init__(self, video_path: str, output_dir: str = "detected_slides",
                 threshold_percent: float = 1.0, min_duration: float = 2.0,
                 check_interval: float = 0.5):
        """
        Initializes the SlideDetector.

        Args:
            video_path (str): Path to the source video file.
            output_dir (str): Directory to save extracted slide images (None to disable saving).
            threshold_percent (float): Percentage of pixel change required to trigger a new slide (0.0 - 100.0).
            min_duration (float): Minimum duration of a slide in seconds to be valid.
            check_interval (float): Interval in seconds between checking frames (higher = faster, lower = more precise).
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold_percent = threshold_percent
        self.min_duration = min_duration
        self.check_interval = check_interval
        self.slides: List[Dict] = []

        # Internal state
        self._cap = None
        self._fps = 0.0
        self._total_frames = 0
        self._duration = 0.0

    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Formats seconds into HH:MM:SS.mmm format.

        Args:
            seconds (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts frame to grayscale and applies blurring to reduce noise.

        Args:
            frame (np.ndarray): The original BGR frame.

        Returns:
            np.ndarray: Processed grayscale frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to smooth out compression artifacts and noise
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def _calculate_change_percentage(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculates the percentage of changed pixels between two frames.

        Args:
            frame1 (np.ndarray): First frame (processed).
            frame2 (np.ndarray): Second frame (processed).

        Returns:
            float: Percentage of pixels changed (0-100).
        """
        # Calculate absolute difference
        frame_delta = cv2.absdiff(frame1, frame2)

        # Threshold the delta to ignore minor light fluctuations
        # Pixels with a change < 25 (out of 255) are ignored
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size

        return (changed_pixels / total_pixels) * 100

    def _save_slide_image(self, frame: np.ndarray, slide_idx: int):
        """
        Saves the current frame as an image file.

        Args:
            frame (np.ndarray): The frame to save.
            slide_idx (int): The index of the slide.
        """
        if self.output_dir:
            filename = os.path.join(self.output_dir, f"slide_{slide_idx:03d}.jpg")
            cv2.imwrite(filename, frame)

    def _initialize_capture(self) -> bool:
        """
        Opens the video capture and reads metadata.

        Returns:
            bool: True if video opened successfully, False otherwise.
        """
        if not os.path.exists(self.video_path):
            print(f"Error: File '{self.video_path}' does not exist.")
            return False

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            print("Error: Could not open video source.")
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps

        print(f"Analyzing video: {self.video_path}")
        print(f"Duration: {self.format_time(self._duration)} ({self._duration:.2f}s), FPS: {self._fps:.2f}")
        return True

    def run(self) -> List[Dict]:
        """
        Runs the slide detection process.

        Returns:
            List[Dict]: A list of detected slides with start/end times.
        """
        if not self._initialize_capture():
            return []

        # Check frame every 0.5 seconds to speed up processing
        step_frames = int(self._fps * self.check_interval)
        if step_frames < 1: step_frames = 1

        # State tracking
        current_slide_start = 0.0
        slide_idx = 1
        self.slides = []

        # Read first frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        if not ret: return []

        prev_frame_processed = self._preprocess_frame(frame)
        self._save_slide_image(frame, slide_idx)

        print(f"\n[Processing...]")

        # Main loop - skipping frames for performance
        pbar = tqdm(range(step_frames, self._total_frames, step_frames),
                   desc="Detected: 0",
                   unit="frames")

        for frame_no in pbar:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, current_frame = self._cap.read()
            if not ret: break

            current_frame_processed = self._preprocess_frame(current_frame)
            change_pct = self._calculate_change_percentage(prev_frame_processed, current_frame_processed)
            current_time = frame_no / self._fps

            # Check if change exceeds threshold
            if change_pct > self.threshold_percent:

                # Ensure minimum duration for the previous slide
                if (current_time - current_slide_start) >= self.min_duration:

                    duration = current_time - current_slide_start

                    # Record previous slide
                    self.slides.append({
                        "id": slide_idx,
                        "start": current_slide_start,
                        "end": current_time,
                        "duration": duration
                    })

                    # Advance to next slide
                    slide_idx += 1
                    current_slide_start = current_time
                    prev_frame_processed = current_frame_processed
                    self._save_slide_image(current_frame, slide_idx)

                    # Update progress bar description with detected count
                    pbar.set_description(f"Detected: {slide_idx-1}")

                else:
                    # Change happened too quickly (animation/transition).
                    # Update reference frame to accumulate changes.
                    prev_frame_processed = current_frame_processed

        pbar.close()

        # Handle the final slide
        final_duration = self._duration - current_slide_start
        self.slides.append({
            "id": slide_idx,
            "start": current_slide_start,
            "end": self._duration,
            "duration": final_duration
        })

        self._cap.release()

        # Print summary at the end
        print("\n" + "="*60)
        for slide in self.slides:
            print(f"Slide {slide['id']}: {self.format_time(slide['start'])} - "
                  f"{self.format_time(slide['end'])} (Duration: {slide['duration']:.2f}s)")
        print("="*60 + "\n")

        return self.slides

if __name__ == "__main__":
    # Example usage
    video_file = "test_video.mp4"

    # Command line argument support
    if len(sys.argv) > 1:
        video_file = sys.argv[1]

    detector = SlideDetector(video_file, output_dir="detected_slides", threshold_percent=2)
    slides = detector.run()
