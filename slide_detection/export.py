"""Result export — representative slide images, OCR text, and slides.json."""
import json
import logging
import os
from typing import Dict, List, Optional

import cv2

from .config import DetectorConfig
from .imaging import format_time
from .video import VideoSource

try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

logger = logging.getLogger("SlideDetector")


class SlideExporter:
    """Saves the sharpest frame for each slide, runs OCR, and writes slides.json."""

    def __init__(self, video: VideoSource, config: DetectorConfig,
                 output_dir: Optional[str]):
        self.video = video
        self.config = config
        self.output_dir = output_dir  # resolved per-video directory, or None
        self.use_ocr = config.use_ocr and _HAS_OCR
        if config.use_ocr and not _HAS_OCR:
            logger.warning("pytesseract/PIL not installed (`pip install pytesseract pillow` "
                           "and a Tesseract binary). OCR disabled.")

    def _best_frame_at_slide(self, slide: Dict):
        """
        Samples 5 frames from the stable middle portion of the slide and returns
        the sharpest one, measured by Laplacian variance of the grayscale image.
        Avoids transition artefacts near the edges of the slide window.
        """
        content_start = slide.get("content_start", slide["start"])
        stable_duration = slide["end"] - content_start
        margin = min(0.5, stable_duration * 0.15)
        t_start = content_start + margin
        t_end = slide["end"] - margin
        if t_end <= t_start:
            t_start = content_start + stable_duration * 0.1
            t_end = slide["end"] - stable_duration * 0.1

        n_samples = 5
        best_frame = None
        best_score = -1.0

        cap = cv2.VideoCapture(self.video.path)
        for k in range(n_samples):
            t = t_start + (t_end - t_start) * k / max(n_samples - 1, 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame
        cap.release()
        return best_frame

    def _ocr_slide_image(self, image_path: str) -> str:
        """
        Runs Tesseract OCR on a saved slide image and returns a single-line
        string with collapsed whitespace. Returns "" on any failure (missing
        binary, unsupported language pack, unreadable image).
        """
        if not self.use_ocr:
            return ""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.config.ocr_lang)
            return " ".join(text.split())
        except Exception:
            return ""

    def export_slide_images(self, slides: List[Dict]) -> None:
        """Saves the sharpest representative frame for each final slide and runs OCR if enabled."""
        if not self.output_dir:
            return
        logger.info("\n[Exporting slide images...]")
        for slide in slides:
            frame = self._best_frame_at_slide(slide)
            if frame is not None:
                filename = f"slide_{slide['id']:03d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame)
                slide["image"] = filename
                slide["ocr_slide_text"] = (self._ocr_slide_image(filepath)
                                 if self.use_ocr and slide["type"] == "slide" else "")
                preview = (slide["ocr_slide_text"][:60] + "...") if len(slide["ocr_slide_text"]) > 60 else slide["ocr_slide_text"]
                logger.info(f"  {filename}  [{format_time(slide['start'])} – {format_time(slide['end'])}]"
                      f"  type={slide['type']}  conf={slide.get('confidence', 1.0):.2f}"
                      + (f"  text={preview!r}" if preview else ""))

    def export_json(self, slides: List[Dict]) -> None:
        """
        Writes slides metadata to slides.json in output_dir.

        Each entry contains id, start, end, duration, type and image filename.
        The 'type' field signals the pipeline how to handle each segment:
          'slide'  → normal presentation slide; paraphrase with compression.
          'camera' → full-screen camera or fast demo; preserve word count (no compression).
        """
        if not self.output_dir:
            return
        path = os.path.join(self.output_dir, "slides.json")
        payload = [
            {
                "id": s["id"],
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "duration": round(s["duration"], 3),
                "type": s["type"],
                "image": s["image"],
                "ocr_slide_text": s.get("ocr_slide_text", ""),
                "confidence": round(float(s.get("confidence", 1.0)), 3),
                "needs_review": bool(s.get("needs_review", False)),
            }
            for s in slides
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"\n[Exported metadata → {path}]")