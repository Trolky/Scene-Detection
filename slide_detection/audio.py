"""Audio extraction and silence-based scoring of slide boundaries."""
import logging
import os
import subprocess
import tempfile
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger("SlideDetector")


class AudioAnalyzer:
    """Extracts mono PCM audio from a video once and computes silence scores around given timestamps.

    Real slide transitions are almost always accompanied by a brief speech pause,
    which can be used to cross-validate visually detected boundaries.
    """

    def __init__(self, audio_sr: int = 16000):
        self.audio_sr = audio_sr
        self._audio_data: Optional[np.ndarray] = None

    @property
    def loaded(self) -> bool:
        """True when audio is loaded and available for scoring."""
        return self._audio_data is not None

    @property
    def seconds(self) -> float:
        """Length of the loaded audio in seconds (0.0 when not loaded)."""
        if self._audio_data is None:
            return 0.0
        return len(self._audio_data) / self.audio_sr

    def load(self, video_path: str) -> bool:
        """
        Extracts mono PCM audio at audio_sr Hz using ffmpeg via subprocess.

        Stores float32 samples in [-1, 1] internally. Returns False when ffmpeg
        is unavailable or the source has no audio track — silence scoring then
        falls back to a neutral 0.5.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ac", "1", "-ar", str(self.audio_sr),
                 "-vn", "-loglevel", "error", tmp.name],
                check=True, capture_output=True,
            )
            with wave.open(tmp.name, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            self._audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        except (subprocess.CalledProcessError, FileNotFoundError, wave.Error) as e:
            logger.warning(f"Audio extraction failed ({type(e).__name__}). "
                           "Audio cross-validation disabled — install ffmpeg and add it to "
                           "PATH to enable.")
            self._audio_data = None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return self._audio_data is not None

    def silence_score_at(self, t: float, window: float = 0.5) -> float:
        """
        Returns a 0..1 score indicating how strongly t coincides with a local
        audio silence. 1.0 = clear pause at t; ~0.2 = no dip at all.

        Computes 50 ms RMS frames in a ±window region around t. The dip ratio
        (min RMS / mean RMS) measures how much the quietest frame stands out
        from the surrounding speech. A small dip ratio → strong silence.

        Returns 0.5 (neutral) when audio is unavailable or the segment is too
        short to estimate reliably.
        """
        if self._audio_data is None:
            return 0.5

        sr = self.audio_sr
        frame_size = int(sr * 0.05)
        half = int(window * sr)
        center = int(t * sr)
        start_idx = max(0, center - half)
        end_idx = min(len(self._audio_data), center + half)

        segment = self._audio_data[start_idx:end_idx]
        n_frames = len(segment) // frame_size
        if n_frames < 4:
            return 0.5

        rms = np.sqrt(np.mean(
            segment[: n_frames * frame_size].reshape(n_frames, frame_size) ** 2,
            axis=1,
        ))
        mean_rms = float(rms.mean())
        if mean_rms < 1e-6:
            return 1.0  # whole region is silent

        dip_ratio = float(rms.min()) / mean_rms
        return float(np.clip(1.0 - dip_ratio, 0.2, 1.0))