"""Deepgram STT — transcribes the full video and returns word-level data."""
import logging
import os
import subprocess
import tempfile
from typing import List, Dict

from deepgram import DeepgramClient, PrerecordedOptions

logger = logging.getLogger("Pipeline.STT")


class DeepgramTranscriber:
    """Transcribes a video file via Deepgram REST API.

    Extracts mono 16 kHz WAV audio with ffmpeg, sends it to Deepgram nova-2,
    and returns a flat list of word dicts with precise timestamps.
    """

    def __init__(self, api_key: str):
        self.client = DeepgramClient(api_key)

    def transcribe(self, video_path: str, language: str = "cs") -> List[Dict]:
        """Returns word-level transcript for the entire video.

        Each element: {"word": str, "start": float, "end": float, "confidence": float}
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._extract_audio(video_path, tmp_path)
            return self._call_deepgram(tmp_path, language)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _extract_audio(self, video_path: str, output_path: str) -> None:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-vn",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                + result.stderr.decode(errors="replace")
            )
        size_mb = os.path.getsize(output_path) / 1_048_576
        logger.info(f"[STT] Audio extracted: {size_mb:.1f} MB WAV @ 16 kHz mono")

    def _call_deepgram(self, audio_path: str, language: str) -> List[Dict]:
        options = PrerecordedOptions(
            model="nova-2",
            language=language,
            smart_format=True,
            punctuate=True,
            utterances=False,
        )

        with open(audio_path, "rb") as f:
            buffer = f.read()

        source = {"buffer": buffer}
        response = self.client.listen.rest.v("1").transcribe_file(source, options)

        alternatives = response.results.channels[0].alternatives
        if not alternatives:
            logger.warning("[STT] Deepgram returned no alternatives — empty transcript")
            return []

        raw_words = alternatives[0].words or []
        words = [
            {
                "word": getattr(w, "punctuated_word", None) or w.word,
                "start": float(w.start),
                "end": float(w.end),
                "confidence": round(float(w.confidence), 4),
            }
            for w in raw_words
        ]
        logger.info(f"[STT] {len(words)} words transcribed")
        return words