"""PipelineRunner — orchestrates STT alignment and paraphrasing."""
import json
import logging
from typing import Dict, List

from .paraphrase import AnthropicParaphraser
from .stt import DeepgramTranscriber

logger = logging.getLogger("Pipeline.Runner")


class PipelineRunner:
    """Runs the full STT → paraphrase pipeline on top of a slides.json.

    Steps:
      1. Transcribe the full video with Deepgram (one API call, word timestamps).
      2. Align words to each segment by timestamp → adds 'transcript' and 'words'.
      3. Paraphrase each slide segment with Anthropic (~50 % compression).
         Camera and demo segments keep the original transcript.
      4. Write the enriched data back to the same slides.json.
    """

    def __init__(
        self,
        video_path: str,
        slides_json_path: str,
        deepgram_api_key: str,
        anthropic_api_key: str,
        compression_ratio: float = 0.5,
        language: str = "cs",
        anthropic_model: str = "claude-haiku-4-5",
    ):
        self.video_path = video_path
        self.slides_json_path = slides_json_path
        self.language = language
        self.transcriber = DeepgramTranscriber(deepgram_api_key)
        self.paraphraser = AnthropicParaphraser(
            anthropic_api_key,
            model=anthropic_model,
            compression_ratio=compression_ratio,
        )

    def run(self) -> List[Dict]:
        """Executes the pipeline and returns the enriched slide list."""
        slides = self._load_slides()

        logger.info(f"[Pipeline] Processing {len(slides)} segments from {self.slides_json_path}")

        logger.info("[Pipeline] Step 1/3 — Deepgram STT")
        words = self.transcriber.transcribe(self.video_path, self.language)

        logger.info("[Pipeline] Step 2/3 — Aligning words to segments")
        self._align_transcripts(slides, words)

        # logger.info("[Pipeline] Step 3/3 — OpenAI paraphrasing")
        # self._paraphrase_slides(slides)

        self._save_slides(slides)
        logger.info(f"[Pipeline] Done → {self.slides_json_path}")
        return slides

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_slides(self) -> List[Dict]:
        with open(self.slides_json_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _ends_sentence(word: str) -> bool:
        return bool(word) and word[-1] in ".!?"

    def _align_transcripts(self, slides: List[Dict], words: List[Dict]) -> None:
        """Assigns 'transcript' to each slide by timestamp overlap.

        A word belongs to a segment when its start falls within [segment.start,
        segment.end). After the initial assignment, each segment's transcript is
        extended to the nearest sentence boundary by borrowing words from the
        following segment (up to 20 words), so the paraphraser never receives a
        mid-sentence fragment.
        """
        # First pass: assign words strictly by timestamp.
        segment_words: List[List[Dict]] = []
        for slide in slides:
            seg_words = [
                w for w in words
                if slide["start"] <= w["start"] < slide["end"]
            ]
            segment_words.append(seg_words)

        # Second pass: extend to sentence boundary where needed.
        for i, slide in enumerate(slides):
            seg_words = segment_words[i]

            if seg_words and not self._ends_sentence(seg_words[-1]["word"]):
                if i + 1 < len(slides):
                    borrowed = []
                    for word in segment_words[i + 1]:
                        borrowed.append(word)
                        if self._ends_sentence(word["word"]):
                            break
                        if len(borrowed) >= 20:
                            borrowed = []
                            break
                    seg_words = seg_words + borrowed

            slide["transcript"] = " ".join(w["word"] for w in seg_words)
            logger.info(
                f"  Slide {slide['id']:3d} [{slide['type']:6s}]: "
                f"{len(seg_words)} words  "
                f"({slide['start']:.1f}s – {slide['end']:.1f}s)"
            )

    def _paraphrase_slides(self, slides: List[Dict]) -> None:
        slide_count = sum(1 for s in slides if s["type"] == "slide")
        processed = 0
        for slide in slides:
            seg_type = slide["type"]
            transcript = slide.get("transcript", "")
            if seg_type == "slide":
                processed += 1
                word_count = len(transcript.split())
                logger.info(
                    f"  Paraphrasing slide {slide['id']} "
                    f"({processed}/{slide_count}, {word_count} words)..."
                )
            slide["paraphrase"] = self.paraphraser.paraphrase(transcript, seg_type)

    def _save_slides(self, slides: List[Dict]) -> None:
        with open(self.slides_json_path, "w", encoding="utf-8") as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)