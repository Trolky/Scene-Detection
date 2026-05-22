"""STT + paraphrase pipeline for lecture video compression.

Reads slides.json produced by SlideDetector, transcribes the full video
with Deepgram, aligns word-level results to each segment, then paraphrases
slide segments with OpenAI (~50 % compression). Enriches slides.json in place.
"""
from .runner import PipelineRunner

__all__ = ["PipelineRunner"]