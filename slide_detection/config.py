"""Detector configuration — all tunable parameters in one place."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable parameters for slide detection.

    Immutable (frozen) — created once in SlideDetector.__init__ and
    shared by all collaborating components (VideoSource, PostProcessor,
    SlideClassifier, ConfidenceScorer, SlideExporter).

    Attributes:
        threshold_percent: Percentage of pixel change required to trigger a
            new slide (0.0 - 100.0).
        min_duration: Minimum duration of a slide in seconds to be valid.
        check_interval: Interval in seconds between checking frames
            (higher = faster, lower = more precise).
        min_changed_blocks: Min number of 4x4 grid blocks that must change for
            a slide transition. Lower values are more sensitive; raise to
            ignore a PiP camera in a corner.
        confirm_transitions: If True, a detected transition is only accepted
            when the frame one check_interval later still differs from the
            pre-transition frame. Filters out single-frame glitches and
            compression artefacts.
        similarity_threshold: Max change % between two slides to consider them
            visually identical.
        camera_segment_min_count: Min number of consecutive short slides in a
            run to collapse the whole run into one 'camera'/'demo' segment.
        use_face_detection: If True, runs Haar Cascade face detection on each
            'slide' segment after all merging. Segments where a face occupies
            more than face_area_threshold of the frame are reclassified as
            'camera'. Catches stable full-screen camera shots that don't
            produce many short transitions and are missed by the run-length
            heuristic.
        face_area_threshold: Minimum ratio of (face area / frame area) to
            trigger camera reclassification. Default 0.15. Fullscreen camera
            faces typically cover 20–50 % of the frame; a face inside a PiP
            box is < 10 %.
        use_ocr: Run Tesseract OCR on the exported slide images.
        ocr_lang: Tesseract language string; requires matching *.traineddata.
        use_audio_validation: Enable silence-based cross-validation of slide
            boundaries via ffmpeg.
        audio_sr: Sample rate (Hz) for audio analysis.
        confidence_threshold: Boundaries scoring below this value get
            needs_review = True.
        output_dir: Base directory for exported slides; None disables all
            saving. The detector creates a per-video sub-directory inside it.
    """

    # --- Transition detection ---
    threshold_percent: float = 1.0
    min_duration: float = 2.0
    check_interval: float = 0.5
    min_changed_blocks: int = 4
    confirm_transitions: bool = True

    # --- Post-processing ---
    similarity_threshold: float = 2.0
    camera_segment_min_count: int = 5
    use_face_detection: bool = True
    face_area_threshold: float = 0.15

    # --- OCR + audio + confidence ---
    use_ocr: bool = True
    ocr_lang: str = "ces+eng"
    use_audio_validation: bool = True
    audio_sr: int = 16000
    confidence_threshold: float = 0.6

    # --- Output ---
    output_dir: Optional[str] = "detected_slides"