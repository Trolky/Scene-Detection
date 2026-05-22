"""SlideDetection — automatic slide detection in lecture videos.

Public API:
    SlideDetector  — main detection orchestrator.
    DetectorConfig — tunable detection parameters.
"""
from .config import DetectorConfig
from .detector import SlideDetector

__all__ = ["SlideDetector", "DetectorConfig"]