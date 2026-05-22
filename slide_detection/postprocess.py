"""Post-processing passes — cleaning up false transitions by merging slides.

Each PostProcessor method is a pure List[Dict] → List[Dict] transformation
(slide list → cleaned and renumbered slide list). Call order is critical —
see the pipeline in SlideDetector.run.
"""
import logging
from typing import Dict, List

from .config import DetectorConfig
from .imaging import change_percentage, format_time
from .video import VideoSource

logger = logging.getLogger("SlideDetector")


def _is_progressive_build(frame_a, frame_b,
                          top_ratio: float = 0.55, top_threshold: float = 1.5,
                          bottom_min_change: float = 2.0) -> bool:
    """
    Returns True when frame_b looks like frame_a with extra content
    revealed in its lower portion (PowerPoint/Keynote progressive build).

    Conditions, all required:
      * Top `top_ratio` of the frames is visually identical
        (change_pct < top_threshold) — heading and earlier bullets unchanged.
      * Bottom region differs (change_pct > bottom_min_change) — a real
        content delta exists, not just identical-frame noise.
      * The bottom of frame_b has higher pixel std than the bottom of
        frame_a — guards against the reverse case where content was
        REMOVED from the bottom (which is not a forward build).

    Frames are expected to be preprocessed (grayscale + Gaussian blur),
    as returned by VideoSource.load_frame_at_time.
    """
    h = frame_a.shape[0]
    cut = int(h * top_ratio)
    top_change = change_percentage(frame_a[:cut], frame_b[:cut])
    if top_change >= top_threshold:
        return False

    bot_a, bot_b = frame_a[cut:], frame_b[cut:]
    bottom_change = change_percentage(bot_a, bot_b)
    if bottom_change < bottom_min_change:
        return False

    return float(bot_b.std()) > float(bot_a.std()) * 1.10


class PostProcessor:
    """Collection of merging passes over the detected slide list."""

    def __init__(self, video: VideoSource, config: DetectorConfig):
        self.video = video
        self.config = config

    def merge_short_slides(self, slides: List[Dict]) -> List[Dict]:
        """
        Post-processing pass: finds slides shorter than min_duration and checks
        whether they visually match the previous or next slide.

        Logic per short slide:
          - Load a frame from the middle of the short slide.
          - Compare with a frame from the middle of the PREVIOUS slide.
          - Compare with a frame from the middle of the NEXT slide.

          Case 1 – similar to PREVIOUS:
              The presenter briefly flipped forward and came back.
              → Absorb the short slide into the previous one (extend its end time).

          Case 2 – similar to NEXT (but not previous):
              The presenter was already moving to the next slide.
              → Absorb the short slide into the next one (extend its start time).

          Case 3 – similar to neither (or no neighbors):
              It really is a distinct, albeit short, slide.
              → Keep it as-is.

        The loop repeats until no more merges occur (handles consecutive short slides).

        Args:
            slides: Initial list of detected slides.

        Returns:
            List[Dict]: Cleaned-up, re-numbered slide list.
        """
        if not slides:
            return slides

        logger.info("\n[Post-processing: checking short slides for false transitions...]")

        merged = True
        while merged:
            merged = False
            new_slides: List[Dict] = []
            i = 0

            while i < len(slides):
                slide = slides[i]

                if slide["duration"] >= self.config.min_duration or slide["type"] == "camera":
                    new_slides.append(slide)
                    i += 1
                    continue

                # --- Short slide: load its representative frame ---
                mid_time = (slide["start"] + slide["end"]) / 2
                short_frame = self.video.load_frame_at_time(mid_time)

                if short_frame is None:
                    # Cannot load frame → keep as-is
                    new_slides.append(slide)
                    i += 1
                    continue

                has_prev = len(new_slides) > 0
                has_next = i + 1 < len(slides)

                # Load neighbor frames
                prev_frame = None
                if has_prev:
                    prev_mid = (new_slides[-1]["start"] + new_slides[-1]["end"]) / 2
                    prev_frame = self.video.load_frame_at_time(prev_mid)

                next_frame = None
                if has_next:
                    next_mid = (slides[i + 1]["start"] + slides[i + 1]["end"]) / 2
                    next_frame = self.video.load_frame_at_time(next_mid)

                similar_to_prev = (
                    prev_frame is not None and
                    change_percentage(short_frame, prev_frame) < self.config.similarity_threshold
                )
                similar_to_next = (
                    next_frame is not None and
                    change_percentage(short_frame, next_frame) < self.config.similarity_threshold
                )
                similar_prev_to_next = (
                    prev_frame is not None and
                    next_frame is not None and
                    change_percentage(prev_frame, next_frame) < self.config.similarity_threshold
                )

                if similar_prev_to_next:
                    # Presenter accidentally switched slide and came back:
                    # prev and next are the same slide → merge all three
                    prev = new_slides[-1]
                    nxt = slides[i + 1]
                    logger.info(f"  Slide {slide['id']} ({slide['duration']:.2f}s): slide {prev['id']} (before) "
                          f"and slide {nxt['id']} (after) match → merging all three into slide {prev['id']}")
                    prev["end"] = nxt["end"]
                    prev["duration"] = prev["end"] - prev["start"]
                    merged = True
                    i += 2  # skip both the short slide and the next slide

                elif similar_to_prev:
                    # Absorb into previous slide
                    prev = new_slides[-1]
                    logger.info(f"  Slide {slide['id']} ({slide['duration']:.2f}s) matches PREVIOUS "
                          f"slide {prev['id']} → merging into slide {prev['id']}")
                    prev["end"] = slide["end"]
                    prev["duration"] = prev["end"] - prev["start"]
                    merged = True
                    i += 1

                elif similar_to_next:
                    # Absorb into next slide (modify it in-place before we process it)
                    nxt = slides[i + 1]
                    logger.info(f"  Slide {slide['id']} ({slide['duration']:.2f}s) matches NEXT "
                          f"slide {nxt['id']} → merging into slide {nxt['id']}")
                    slides[i + 1] = {**nxt, "start": slide["start"],
                                     "duration": nxt["end"] - slide["start"]}
                    merged = True
                    i += 1  # skip the short slide; next iteration will process the updated next slide

                else:
                    # Unique content but too short to be a usable chunk → absorb into neighbour.
                    # Prefer merging forward (presenter is moving ahead); fall back to previous
                    # only when there is no next slide.
                    # Note: the accidental-click-and-return case (prev ≈ next) is already
                    # handled above by similar_prev_to_next, so it never reaches this branch.
                    if has_next:
                        nxt = slides[i + 1]
                        logger.info(f"  Slide {slide['id']} ({slide['duration']:.2f}s) is unique but too short "
                              f"→ merging into next slide {nxt['id']}")
                        slides[i + 1] = {**nxt, "start": slide["start"],
                                         "duration": nxt["end"] - slide["start"]}
                        merged = True
                        i += 1
                    elif has_prev:
                        prev = new_slides[-1]
                        logger.info(f"  Slide {slide['id']} ({slide['duration']:.2f}s) is unique but too short "
                              f"→ merging into previous slide {prev['id']}")
                        prev["end"] = slide["end"]
                        prev["duration"] = prev["end"] - prev["start"]
                        merged = True
                        i += 1
                    else:
                        new_slides.append(slide)
                        i += 1

            slides = new_slides

        # Re-number IDs sequentially
        for idx, slide in enumerate(slides, start=1):
            slide["id"] = idx

        return slides

    def merge_camera_segments(self, slides: List[Dict]) -> List[Dict]:
        """
        Merges runs of consecutive short slides into a single slide.

        After merge_short_slides, any remaining cluster of >= camera_segment_min_count
        consecutive slides shorter than min_duration is collapsed into one slide spanning
        the entire run. This handles both full-screen camera feeds (continuous motion)
        and fast-changing demos (rapid image switches).

        Args:
            slides: Slide list after merge post-processing.

        Returns:
            List[Dict]: Slide list with camera/demo segments merged and IDs re-numbered.
        """
        if self.config.camera_segment_min_count <= 0:
            return slides

        logger.info("\n[Post-processing: merging consecutive short-slide runs...]")
        result: List[Dict] = []
        i = 0
        while i < len(slides):
            if slides[i]["duration"] < self.config.min_duration:
                # Walk forward to find the extent of this short-slide run
                j = i
                while j < len(slides) and slides[j]["duration"] < self.config.min_duration:
                    j += 1
                count = j - i
                if count >= self.config.camera_segment_min_count:
                    merged_start = slides[i]["start"]
                    merged_end = slides[j - 1]["end"]
                    seg_start = format_time(merged_start)
                    seg_end = format_time(merged_end)
                    logger.info(f"  Merged segment: {count} consecutive short slides "
                          f"[{seg_start} – {seg_end}] → 1 slide")
                    result.append({
                        "id": 0,
                        "start": merged_start,
                        "end": merged_end,
                        "duration": merged_end - merged_start,
                        "type": "slide",
                        "image": None,
                        "content_start": merged_start,
                    })
                else:
                    result.extend(slides[i:j])
                i = j
            else:
                result.append(slides[i])
                i += 1

        for idx, slide in enumerate(result, start=1):
            slide["id"] = idx

        return result

    def merge_consecutive_noncontent(self, slides: List[Dict]) -> List[Dict]:
        """
        Merges consecutive segments of the same non-content type (camera or demo)
        into one. Camera and demo are kept separate — they are visually distinct.
        """
        result: List[Dict] = []
        i = 0
        while i < len(slides):
            seg_type = slides[i]["type"]
            if seg_type not in ("camera", "demo"):
                result.append(slides[i])
                i += 1
                continue
            j = i + 1
            while j < len(slides) and slides[j]["type"] == seg_type:
                j += 1
            if j > i + 1:
                merged_start = slides[i]["start"]
                merged_end = slides[j - 1]["end"]
                logger.info(f"  Merging {j - i} consecutive {seg_type} segments "
                      f"[{format_time(merged_start)} – {format_time(merged_end)}]")
                result.append({
                    "id": 0,
                    "start": merged_start,
                    "end": merged_end,
                    "duration": merged_end - merged_start,
                    "type": seg_type,
                    "image": None,
                    "content_start": slides[i].get("content_start", merged_start),
                })
            else:
                result.append(slides[i])
            i = j
        for idx, slide in enumerate(result, start=1):
            slide["id"] = idx
        return result

    def merge_progressive_builds(self, slides: List[Dict]) -> List[Dict]:
        """
        Merges adjacent 'slide' pairs that look like a progressive build —
        a single slide revealing bullets one at a time. Without this pass
        each reveal step is detected as a separate slide, fragmenting the
        downstream audio chunk and pointing each chunk at an incomplete
        thumbnail.

        The merged segment keeps the LATER slide's representative frame
        (it's the most complete version of the slide) and extends start
        backward to the first reveal step's start. Loops until no further
        merges occur — handles N-step builds.
        """
        if not slides:
            return slides

        logger.info("\n[Post-processing: merging progressive builds...]")
        merged = True
        while merged:
            merged = False
            new_slides: List[Dict] = []
            i = 0
            while i < len(slides):
                if (i + 1 < len(slides)
                        and slides[i]["type"] == "slide"
                        and slides[i + 1]["type"] == "slide"):
                    f1 = self.video.load_frame_at_time(
                        (slides[i].get("content_start", slides[i]["start"]) + slides[i]["end"]) / 2
                    )
                    f2 = self.video.load_frame_at_time(
                        (slides[i + 1].get("content_start", slides[i + 1]["start"]) + slides[i + 1]["end"]) / 2
                    )
                    if (f1 is not None and f2 is not None
                            and _is_progressive_build(f1, f2)):
                        logger.info(f"  Slides {slides[i]['id']} → {slides[i + 1]['id']} "
                                    f"look like a progressive build → merging "
                                    f"(keeping slide {slides[i + 1]['id']}'s frame)")
                        new_slides.append({
                            **slides[i + 1],
                            "start": slides[i]["start"],
                            "duration": slides[i + 1]["end"] - slides[i]["start"],
                        })
                        merged = True
                        i += 2
                        continue
                new_slides.append(slides[i])
                i += 1
            slides = new_slides

        for idx, slide in enumerate(slides, start=1):
            slide["id"] = idx
        return slides

    def merge_similar_adjacent(self, slides: List[Dict]) -> List[Dict]:
        """
        Final deduplication pass: merges adjacent 'slide' pairs that are visually
        identical (change < similarity_threshold). Handles false transitions caused
        by minor on-screen events (cursor, tooltip, brief annotation).
        Repeats until no further merges occur.
        """
        if not slides:
            return slides

        logger.info("\n[Post-processing: merging visually identical adjacent slides...]")
        merged = True
        while merged:
            merged = False
            new_slides: List[Dict] = []
            i = 0
            while i < len(slides):
                if (i + 1 < len(slides)
                        and slides[i]["type"] == "slide"
                        and slides[i + 1]["type"] == "slide"):
                    f1 = self.video.load_frame_at_time(
                        (slides[i].get("content_start", slides[i]["start"]) + slides[i]["end"]) / 2
                    )
                    f2 = self.video.load_frame_at_time(
                        (slides[i + 1].get("content_start", slides[i + 1]["start"]) + slides[i + 1]["end"]) / 2
                    )
                    if (f1 is not None and f2 is not None
                            and change_percentage(f1, f2) < self.config.similarity_threshold):
                        logger.info(f"  Slides {slides[i]['id']} and {slides[i + 1]['id']} "
                              f"are visually identical → merging")
                        new_slides.append({
                            **slides[i],
                            "end": slides[i + 1]["end"],
                            "duration": slides[i + 1]["end"] - slides[i]["start"],
                        })
                        merged = True
                        i += 2
                        continue
                new_slides.append(slides[i])
                i += 1
            slides = new_slides

        for idx, slide in enumerate(slides, start=1):
            slide["id"] = idx
        return slides