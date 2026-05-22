"""Anthropic paraphraser — compresses slide transcripts to ~50 % word count."""
import logging
import anthropic

logger = logging.getLogger("Pipeline.Paraphrase")

_SYSTEM_PROMPT = (
    "Jsi asistent pro kompresaci přednáškových přepisů. "
    "Tvým úkolem je parafrázovat daný přepis na přibližně {ratio} % původního počtu slov v češtině. "
    "Zachovej klíčové informace a fakta. "
    "Odstraň filler slova (um, em, tak, prostě, vlastně, zkrátka), zbytečná opakování "
    "a přehnaně rozvleklé formulace. "
    "Výstup musí být plynný, přirozený český text — žádné komentáře ani vysvětlení navíc."
)


class AnthropicParaphraser:
    """Paraphrases Czech lecture transcripts using the Anthropic API.

    Slide segments are compressed to ~compression_ratio of the original word
    count. Camera and demo segments are returned unchanged — word-count
    preservation is required there so TTS timing matches the original video.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5",
        compression_ratio: float = 0.5,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.compression_ratio = compression_ratio
        self._system_prompt = _SYSTEM_PROMPT.format(ratio=int(compression_ratio * 100))

    def paraphrase(self, text: str, segment_type: str) -> str:
        """Returns compressed text for 'slide' segments; original for camera/demo."""
        if segment_type != "slide":
            return text

        text = text.strip()
        if not text:
            return text

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._system_prompt,
                messages=[
                    {"role": "user", "content": text},
                ],
            )
            result = response.content[0].text.strip()
            original_words = len(text.split())
            result_words = len(result.split())
            logger.debug(
                f"[Paraphrase] {original_words} → {result_words} words "
                f"({result_words / max(original_words, 1):.0%})"
            )
            return result
        except anthropic.APIError as exc:
            logger.warning(f"[Paraphrase] Anthropic call failed: {exc} — keeping original")
            return text