"""SlideDetector — slide detection orchestrator.

Wires together the collaborating components (VideoSource, AudioAnalyzer,
PostProcessor, SlideClassifier, ConfidenceScorer, SlideExporter), drives
the detection loop, and orders the post-processing passes.
"""
import logging
import os
import sys
from typing import Dict, List, Optional

from tqdm import tqdm

from .audio import AudioAnalyzer
from .classify import SlideClassifier
from .confidence import ConfidenceScorer
from .config import DetectorConfig
from .export import SlideExporter
from .imaging import change_percentage, count_changed_blocks, format_time, preprocess_frame
from .postprocess import PostProcessor
from .video import VideoSource

logger = logging.getLogger("SlideDetector")


class SlideDetector:
    """Detects presentation slides in a video file by analyzing frame differences.

    Accepts the same keyword arguments as before — internally builds an
    immutable :class:`DetectorConfig` shared with all components.
    See DetectorConfig docstring for parameter descriptions.
    """

    def __init__(self, video_path: str, output_dir: str = "detected_slides",
                 threshold_percent: float = 1.0, min_duration: float = 2.0,
                 check_interval: float = 0.5, similarity_threshold: float = 2.0,
                 min_changed_blocks: int = 4, camera_segment_min_count: int = 5,
                 confirm_transitions: bool = True, use_face_detection: bool = True,
                 face_area_threshold: float = 0.15,
                 use_ocr: bool = True, ocr_lang: str = "ces+eng",
                 use_audio_validation: bool = True, audio_sr: int = 16000,
                 confidence_threshold: float = 0.6):
        self.video_path = video_path
        self.config = DetectorConfig(
            threshold_percent=threshold_percent,
            min_duration=min_duration,
            check_interval=check_interval,
            min_changed_blocks=min_changed_blocks,
            confirm_transitions=confirm_transitions,
            similarity_threshold=similarity_threshold,
            camera_segment_min_count=camera_segment_min_count,
            use_face_detection=use_face_detection,
            face_area_threshold=face_area_threshold,
            use_ocr=use_ocr,
            ocr_lang=ocr_lang,
            use_audio_validation=use_audio_validation,
            audio_sr=audio_sr,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
        )
        # Resolved per-video output directory; filled in by run().
        self.output_dir: Optional[str] = output_dir
        self.slides: List[Dict] = []

    @staticmethod
    def format_time(seconds: float) -> str:
        """Formats seconds into HH:MM:SS.mmm format. Kept for backward compatibility."""
        return format_time(seconds)

    def _resolve_output_dir(self) -> Optional[str]:
        """Builds and creates the per-video output sub-directory (None to disable saving)."""
        base = self.config.output_dir
        if not base:
            return None
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        resolved = os.path.join(base, video_name)
        os.makedirs(resolved, exist_ok=True)
        return resolved

    def _setup_logging(self) -> None:
        """
        Wires up a per-video FileHandler that writes detection.log next to the
        exported slides. Removes any FileHandler from a previous run to keep
        logs from bleeding between videos when the same process processes
        many files in a loop. A single shared StreamHandler (stdout) is also
        attached the first time so progress stays visible interactively.
        """
        logger.setLevel(logging.INFO)

        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()

        if self.output_dir:
            log_path = os.path.join(self.output_dir, "detection.log")
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)
                   for h in logger.handlers):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(stream_handler)

    def _confirm_transition(self, video: VideoSource, prev_frame, current_time: float) -> bool:
        """
        Verifies a detected transition by sampling the frame one check_interval later.

        Returns True (transition confirmed) when that frame still differs significantly
        from prev_frame, meaning the new scene persisted. Returns False when the image
        returned to something similar to prev_frame, indicating a glitch or artefact.
        """
        confirm_time = current_time + self.config.check_interval
        if confirm_time >= video.duration:
            return True
        confirm_frame = video.load_frame_at_time(confirm_time)
        if confirm_frame is None:
            return True
        return change_percentage(prev_frame, confirm_frame) > self.config.threshold_percent

    def _detect_raw_slides(self, video: VideoSource) -> List[Dict]:
        """
        Runs the core detection loop and returns the raw (pre-post-processing)
        slide list. Each detected transition splits a new slide; short ones are
        kept here and cleaned up later by the post-processing passes.
        """
        cfg = self.config
        logger.info("\n[Processing...]")

        frames = video.iter_sampled_frames(cfg.check_interval)
        try:
            _, first_frame = next(frames)
        except StopIteration:
            return []

        prev_frame_processed = preprocess_frame(first_frame)
        current_slide_start = 0.0
        slide_idx = 1
        slides: List[Dict] = []

        pbar = tqdm(total=video.sample_count(cfg.check_interval),
                    desc="Detected: 0", unit="checks")

        for current_time, current_frame in frames:
            pbar.update(1)

            current_frame_processed = preprocess_frame(current_frame)
            change_pct = change_percentage(prev_frame_processed, current_frame_processed)

            changed_blocks = (
                count_changed_blocks(prev_frame_processed, current_frame_processed)
                if change_pct > cfg.threshold_percent else 0
            )

            if change_pct > cfg.threshold_percent and changed_blocks >= cfg.min_changed_blocks:
                if cfg.confirm_transitions and not self._confirm_transition(
                        video, prev_frame_processed, current_time):
                    # Scene reverted one interval later → single-frame glitch, ignore
                    prev_frame_processed = current_frame_processed
                else:
                    # Record slide regardless of duration – post-processing handles short ones
                    slides.append({
                        "id": slide_idx,
                        "start": current_slide_start,
                        "end": current_time,
                        "duration": current_time - current_slide_start,
                        "type": "slide",
                        "image": None,
                        "content_start": current_slide_start,
                    })

                    slide_idx += 1
                    current_slide_start = current_time
                    prev_frame_processed = current_frame_processed
                    pbar.set_description(f"Detected: {slide_idx - 1}")
            else:
                prev_frame_processed = current_frame_processed

        pbar.close()

        # Final slide
        slides.append({
            "id": slide_idx,
            "start": current_slide_start,
            "end": video.duration,
            "duration": video.duration - current_slide_start,
            "type": "slide",
            "image": None,
            "content_start": current_slide_start,
        })
        return slides

    def run(self) -> List[Dict]:
        """
        Runs the slide detection process.

        Returns:
            List[Dict]: A list of detected slides with start/end times.
        """
        if not os.path.exists(self.video_path):
            logger.error(f"File '{self.video_path}' does not exist.")
            return []

        self.output_dir = self._resolve_output_dir()
        self._setup_logging()
        if self.output_dir:
            logger.info(f"Output directory: {self.output_dir}")

        video = VideoSource(self.video_path)
        if not video.open():
            return []

        logger.info(f"Analyzing video: {self.video_path}")
        logger.info(f"Duration: {format_time(video.duration)} ({video.duration:.2f}s), "
                    f"FPS: {video.fps:.2f}")

        audio = AudioAnalyzer(self.config.audio_sr)
        if self.config.use_audio_validation:
            logger.info("[Loading audio for cross-validation...]")
            if audio.load(self.video_path):
                logger.info(f"  Loaded {audio.seconds:.1f}s of audio "
                      f"@ {self.config.audio_sr} Hz")

        # --- Core detection ---
        self.slides = self._detect_raw_slides(video)
        video.release()
        if not self.slides:
            return []

        # --- Collaborators for the post-processing pipeline ---
        postproc = PostProcessor(video, self.config)
        classifier = SlideClassifier(video, self.config)
        scorer = ConfidenceScorer(video, audio, self.config)
        exporter = SlideExporter(video, self.config, self.output_dir)

        # Post-process: merge clusters of short slides (camera feed or fast demo) into one slide.
        # Must run BEFORE merge_short_slides – once that step absorbs short slides into longer
        # neighbours they exceed min_duration and the camera detection misses the whole run.
        self.slides = postproc.merge_camera_segments(self.slides)

        # Post-process: merge remaining isolated short slides that match a neighbour
        self.slides = postproc.merge_short_slides(self.slides)

        # Post-process: reclassify stable camera segments missed by duration-based detection
        if self.config.use_face_detection:
            self.slides = classifier.reclassify_by_face(self.slides)

        # Post-process: reclassify slides that show a demo/IDE overlay TODO not good enough yet, needs improvement
        # self.slides = classifier.reclassify_demo_slides(self.slides)

        # Post-process: merge consecutive segments of the same non-content type (camera / demo)
        self.slides = postproc.merge_consecutive_noncontent(self.slides)

        # Post-process: collapse PowerPoint-style progressive builds (bullet-by-bullet reveals)
        self.slides = postproc.merge_progressive_builds(self.slides)

        # Post-process: merge visually identical adjacent slides (false transitions)
        self.slides = postproc.merge_similar_adjacent(self.slides)

        # Score each final boundary (visual delta + audio silence) → confidence + review flag
        self.slides = scorer.annotate(self.slides)

        # Export final slide images (after merging, with correct numbering); runs OCR per image
        exporter.export_slide_images(self.slides)

        # Export JSON metadata for downstream pipeline consumption
        exporter.export_json(self.slides)

        logger.info("\n" + "=" * 60)
        for slide in self.slides:
            flag = " [REVIEW]" if slide.get("needs_review") else ""
            logger.info(f"Slide {slide['id']} [{slide['type']:6s}]: {format_time(slide['start'])} - "
                  f"{format_time(slide['end'])} ({slide['duration']:.2f}s) "
                  f"conf={slide.get('confidence', 1.0):.2f}{flag}")
        logger.info("=" * 60 + "\n")

        return self.slides