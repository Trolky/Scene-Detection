"""CLI entry point for the STT + paraphrase pipeline.

Usage:
    python run_pipeline.py <video_path> <slides_json_path>

Example:
    python run_pipeline.py videos/lecture.mp4 detected_slides/lecture/slides.json

Reads DEEPGRAM_API_KEY and ANTHROPIC_API_KEY from the environment or a .env file
in the project root.
"""
import logging
import os
import sys

from dotenv import load_dotenv

from pipeline import PipelineRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

VIDEO_PATH = "videos/test_video_1.mp4"
SLIDES_JSON = "detected_slides/test_video_1/slides.json"
ANTHROPIC_MODEL = "claude-haiku-4-5"
COMPRESSION_RATIO = 0.5
LANGUAGE = "cs"


def main() -> None:
    video_path = VIDEO_PATH
    slides_json_path = SLIDES_JSON

    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)
    if not os.path.exists(slides_json_path):
        print(f"Error: slides.json not found: {slides_json_path}")
        sys.exit(1)

    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not deepgram_key:
        print("Error: DEEPGRAM_API_KEY not set in environment or .env")
        sys.exit(1)
    # if not anthropic_key:
    #     print("Error: ANTHROPIC_API_KEY not set in environment or .env")
    #     sys.exit(1)

    runner = PipelineRunner(
        video_path=video_path,
        slides_json_path=slides_json_path,
        deepgram_api_key=deepgram_key,
        anthropic_api_key=anthropic_key,
        compression_ratio=COMPRESSION_RATIO,
        language=LANGUAGE,
        anthropic_model=ANTHROPIC_MODEL,
    )
    slides = runner.run()
    print(f"\nProcessed {len(slides)} segments.")
    slide_count = sum(1 for s in slides if s["type"] == "slide")
    camera_count = sum(1 for s in slides if s["type"] != "slide")
    print(f"  {slide_count} slide segments paraphrased")
    print(f"  {camera_count} camera/demo segments kept verbatim")


if __name__ == "__main__":
    main()