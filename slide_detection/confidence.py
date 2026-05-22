"""Confidence scoring of detected boundaries (visual delta + audio silence)."""
import logging
from typing import Dict, List

import numpy as np

from .audio import AudioAnalyzer
from .config import DetectorConfig
from .imaging import change_percentage, count_changed_blocks, format_time
from .video import VideoSource

logger = logging.getLogger("SlideDetector")


class ConfidenceScorer:
    """Computes a confidence score 0..1 and a needs_review flag for each slide boundary.

    Combines the visual jump between frames around the boundary with the audio silence score.
    """

    def __init__(self, video: VideoSource, audio: AudioAnalyzer, config: DetectorConfig):
        self.video = video
        self.audio = audio
        self.config = config

    def _boundary_confidence(self, t: float) -> Dict[str, float]:
        """
        Combines visual delta and audio silence into a confidence score that
        a real slide transition occurred at time t. Returns sub-scores for
        transparency in the JSON output.

        Visual: change percentage and changed-block count between the frames
        one check_interval before and after t. Strong transitions easily
        exceed change_pct = 20 % and saturate ~12/16 blocks.
        Audio: silence dip ratio in a ±0.5 s window around t.
        Combined: 0.6 · visual + 0.4 · audio (visual is the primary signal).
        """
        eps = self.config.check_interval
        f_before = self.video.load_frame_at_time(max(0.0, t - eps))
        f_after = self.video.load_frame_at_time(min(self.video.duration, t + eps))

        if f_before is None or f_after is None:
            visual_score = 0.5
        else:
            change_pct = change_percentage(f_before, f_after)
            blocks = count_changed_blocks(f_before, f_after)
            visual_score = float(np.clip(
                0.5 * (change_pct / 20.0) + 0.5 * (blocks / 16.0),
                0.0, 1.0,
            ))

        audio_score = self.audio.silence_score_at(t) if self.config.use_audio_validation else 0.5
        combined = 0.6 * visual_score + 0.4 * audio_score
        return {"visual": round(visual_score, 3),
                "audio": round(audio_score, 3),
                "combined": round(combined, 3)}

    def annotate(self, slides: List[Dict]) -> List[Dict]:
        """
        Computes a confidence score for each slide based on the strength of
        its start-of-slide transition (visual delta + audio silence). Sets
        'needs_review' True when confidence < confidence_threshold so a
        downstream UI can quickly surface uncertain boundaries.

        The first slide always gets confidence 1.0 — its start at t = 0 is
        not the result of detection.
        """
        logger.info("\n[Post-processing: scoring boundary confidence...]")
        flagged = 0
        for idx, slide in enumerate(slides):
            if idx == 0:
                slide["confidence"] = 1.0
                slide["confidence_breakdown"] = {"visual": 1.0, "audio": 1.0, "combined": 1.0}
            else:
                scores = self._boundary_confidence(slide["start"])
                slide["confidence"] = scores["combined"]
                slide["confidence_breakdown"] = scores
            slide["needs_review"] = slide["confidence"] < self.config.confidence_threshold
            if slide["needs_review"]:
                flagged += 1
                cb = slide["confidence_breakdown"]
                logger.info(f"  Slide {slide['id']} [{format_time(slide['start'])}] "
                      f"confidence={slide['confidence']:.2f} "
                      f"(visual={cb['visual']:.2f}, audio={cb['audio']:.2f}) → review")
        logger.info(f"  {flagged}/{len(slides)} slides flagged for manual review "
              f"(threshold={self.config.confidence_threshold})")
        return slides