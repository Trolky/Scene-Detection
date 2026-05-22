"""CLI entry point for SlideDetection.

Implementation lives in the slide_detection package. This file just runs
detection on a given video. from main import SlideDetector remains valid
for backward compatibility via the re-export below.
"""
import sys

from slide_detection import SlideDetector

__all__ = ["SlideDetector"]


if __name__ == "__main__":
    # Batch processing: list multiple paths, or pass one as a CLI argument.
    # video_files = ["videos/test_video_1.mp4", "videos/test_video_2.mp4",
    #                "videos/test_video_3.mp4", "videos/test_video_4.mp4",
    #                "videos/test_video_5.mp4"]
    video_files = ["videos/test_video_1.mp4"]
    if len(sys.argv) > 1:
        video_files = [sys.argv[1]]

    for video_file in video_files:
        detector = SlideDetector(
            video_file,
            output_dir="detected_slides",
            threshold_percent=5,
            min_duration=10,
            similarity_threshold=5,
            min_changed_blocks=4,        # ignore changes in < 4/16 blocks (e.g. PiP camera corner)
            camera_segment_min_count=5,  # remove runs of >=5 consecutive short slides (fullscreen camera)
            face_area_threshold=0.15,
            use_ocr=True,                # OCR slide text for STT-transcript matching
            ocr_lang="ces+eng",          # Czech + English; install with `tesseract --list-langs`
            use_audio_validation=True,   # require ffmpeg in PATH
            confidence_threshold=0.6,    # boundaries below this score get needs_review=True
        )
        slides = detector.run()