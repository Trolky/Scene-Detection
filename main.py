import cv2
import json
import numpy as np
import os
import sys
from typing import List, Dict, Optional
from tqdm import tqdm
import time

class SlideDetector:
    """
    Detects presentation slides in a video file by analyzing frame differences.
    """

    def __init__(self, video_path: str, output_dir: str = "detected_slides",
                 threshold_percent: float = 1.0, min_duration: float = 2.0,
                 check_interval: float = 0.5, similarity_threshold: float = 2.0,
                 min_changed_blocks: int = 4, camera_segment_min_count: int = 5,
                 confirm_transitions: bool = True, use_face_detection: bool = True,
                 face_area_threshold: float = 0.15):
        """
        Initializes the SlideDetector.

        Args:
            video_path (str): Path to the source video file.
            output_dir (str): Directory to save extracted slide images (None to disable saving).
            threshold_percent (float): Percentage of pixel change required to trigger a new slide (0.0 - 100.0).
            min_duration (float): Minimum duration of a slide in seconds to be valid.
            check_interval (float): Interval in seconds between checking frames (higher = faster, lower = more precise).
            similarity_threshold (float): Max change % between two slides to consider them visually identical.
            min_changed_blocks (int): Min number of 4x4 grid blocks that must change for a slide transition.
                                      Lower values are more sensitive; raise to ignore a PiP camera in a corner.
            confirm_transitions (bool): If True, a detected transition is only accepted when the frame
                                        one check_interval later still differs from the pre-transition
                                        frame. Filters out single-frame glitches and compression artefacts.
            use_face_detection (bool): If True, runs Haar Cascade face detection on each 'slide' segment
                                       after all merging. Segments where a face occupies more than
                                       face_area_threshold of the frame are reclassified as 'camera'.
                                       Catches stable full-screen camera shots that don't produce
                                       many short transitions and are missed by camera_segment_min_count.
            face_area_threshold (float): Minimum ratio of (face area / frame area) to trigger camera
                                         reclassification. Default 0.15. Fullscreen camera faces typically
                                         cover 20–50 % of the frame; a face inside a PiP box is < 10 %.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold_percent = threshold_percent
        self.min_duration = min_duration
        self.check_interval = check_interval
        self.similarity_threshold = similarity_threshold
        self.min_changed_blocks = min_changed_blocks
        self.camera_segment_min_count = camera_segment_min_count
        self.confirm_transitions = confirm_transitions
        self.use_face_detection = use_face_detection
        self.face_area_threshold = face_area_threshold
        self.slides: List[Dict] = []

        # Internal state
        self._cap = None
        self._fps = 0.0
        self._total_frames = 0
        self._duration = 0.0
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None
        if use_face_detection:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )

    @staticmethod
    def format_time(seconds: float) -> str:
        """Formats seconds into HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Converts frame to grayscale and applies blurring to reduce noise."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def _calculate_change_percentage(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculates the percentage of changed pixels between two frames."""
        frame_delta = cv2.absdiff(frame1, frame2)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        return (changed_pixels / total_pixels) * 100

    def _count_changed_blocks(self, frame1: np.ndarray, frame2: np.ndarray, grid: int = 4) -> int:
        """
        Divides frames into a grid and counts how many cells have significant change.

        A PiP camera occupies only 1–2 cells; a real slide transition spreads across many.

        Args:
            frame1, frame2: Preprocessed (grayscale blurred) frames.
            grid: Number of rows/columns (default 4 → 4×4 = 16 cells).

        Returns:
            Number of cells with change > 5 % of their pixels.
        """
        frame_delta = cv2.absdiff(frame1, frame2)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        h, w = thresh.shape
        cell_h, cell_w = h // grid, w // grid
        changed = 0
        for r in range(grid):
            for c in range(grid):
                cell = thresh[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                if np.count_nonzero(cell) / cell.size * 100 > 5.0:
                    changed += 1
        return changed

    def _confirm_transition(self, prev_frame: np.ndarray, current_time: float) -> bool:
        """
        Verifies a detected transition by sampling the frame one check_interval later.

        Returns True (transition confirmed) when that frame still differs significantly
        from prev_frame, meaning the new scene persisted. Returns False when the image
        returned to something similar to prev_frame, indicating a glitch or artefact.
        """
        confirm_time = current_time + self.check_interval
        if confirm_time >= self._duration:
            return True
        confirm_frame = self._load_frame_at_time(confirm_time)
        if confirm_frame is None:
            return True
        return self._calculate_change_percentage(prev_frame, confirm_frame) > self.threshold_percent

    def _initialize_capture(self) -> bool:
        """Opens the video capture and reads metadata."""
        if not os.path.exists(self.video_path):
            print(f"Error: File '{self.video_path}' does not exist.")
            return False

        if self.output_dir:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            self.output_dir = os.path.join(self.output_dir, video_name)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"Created output directory: {self.output_dir}")

        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            print("Error: Could not open video source.")
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps

        print(f"Analyzing video: {self.video_path}")
        print(f"Duration: {self.format_time(self._duration)} ({self._duration:.2f}s), FPS: {self._fps:.2f}")
        return True

    def _load_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Loads and preprocesses a single frame at a given timestamp.

        Args:
            timestamp (float): Time in seconds.

        Returns:
            Optional[np.ndarray]: Preprocessed frame, or None on failure.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return self._preprocess_frame(frame)

    def _merge_short_slides(self, slides: List[Dict]) -> List[Dict]:
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

        print("\n[Post-processing: checking short slides for false transitions...]")

        merged = True
        while merged:
            merged = False
            new_slides: List[Dict] = []
            i = 0

            while i < len(slides):
                slide = slides[i]

                if slide["duration"] >= self.min_duration or slide["type"] == "camera":
                    new_slides.append(slide)
                    i += 1
                    continue

                # --- Short slide: load its representative frame ---
                mid_time = (slide["start"] + slide["end"]) / 2
                short_frame = self._load_frame_at_time(mid_time)

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
                    prev_frame = self._load_frame_at_time(prev_mid)

                next_frame = None
                if has_next:
                    next_mid = (slides[i + 1]["start"] + slides[i + 1]["end"]) / 2
                    next_frame = self._load_frame_at_time(next_mid)

                similar_to_prev = (
                    prev_frame is not None and
                    self._calculate_change_percentage(short_frame, prev_frame) < self.similarity_threshold
                )
                similar_to_next = (
                    next_frame is not None and
                    self._calculate_change_percentage(short_frame, next_frame) < self.similarity_threshold
                )
                similar_prev_to_next = (
                    prev_frame is not None and
                    next_frame is not None and
                    self._calculate_change_percentage(prev_frame, next_frame) < self.similarity_threshold
                )

                if similar_prev_to_next:
                    # Presenter accidentally switched slide and came back:
                    # prev and next are the same slide → merge all three
                    prev = new_slides[-1]
                    nxt = slides[i + 1]
                    print(f"  Slide {slide['id']} ({slide['duration']:.2f}s): slide {prev['id']} (before) "
                          f"and slide {nxt['id']} (after) match → merging all three into slide {prev['id']}")
                    prev["end"] = nxt["end"]
                    prev["duration"] = prev["end"] - prev["start"]
                    merged = True
                    i += 2  # skip both the short slide and the next slide

                elif similar_to_prev:
                    # Absorb into previous slide
                    prev = new_slides[-1]
                    print(f"  Slide {slide['id']} ({slide['duration']:.2f}s) matches PREVIOUS "
                          f"slide {prev['id']} → merging into slide {prev['id']}")
                    prev["end"] = slide["end"]
                    prev["duration"] = prev["end"] - prev["start"]
                    merged = True
                    i += 1

                elif similar_to_next:
                    # Absorb into next slide (modify it in-place before we process it)
                    nxt = slides[i + 1]
                    print(f"  Slide {slide['id']} ({slide['duration']:.2f}s) matches NEXT "
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
                        print(f"  Slide {slide['id']} ({slide['duration']:.2f}s) is unique but too short "
                              f"→ merging into next slide {nxt['id']}")
                        slides[i + 1] = {**nxt, "start": slide["start"],
                                         "duration": nxt["end"] - slide["start"]}
                        merged = True
                        i += 1
                    elif has_prev:
                        prev = new_slides[-1]
                        print(f"  Slide {slide['id']} ({slide['duration']:.2f}s) is unique but too short "
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

    def _merge_camera_segments(self, slides: List[Dict]) -> List[Dict]:
        """
        Merges runs of consecutive short slides into a single slide.

        After _merge_short_slides, any remaining cluster of >= camera_segment_min_count
        consecutive slides shorter than min_duration is collapsed into one slide spanning
        the entire run. This handles both full-screen camera feeds (continuous motion)
        and fast-changing demos (rapid image switches).

        Args:
            slides: Slide list after merge post-processing.

        Returns:
            List[Dict]: Slide list with camera/demo segments merged and IDs re-numbered.
        """
        if self.camera_segment_min_count <= 0:
            return slides

        print("\n[Post-processing: merging consecutive short-slide runs...]")
        result: List[Dict] = []
        i = 0
        while i < len(slides):
            if slides[i]["duration"] < self.min_duration:
                # Walk forward to find the extent of this short-slide run
                j = i
                while j < len(slides) and slides[j]["duration"] < self.min_duration:
                    j += 1
                count = j - i
                if count >= self.camera_segment_min_count:
                    merged_start = slides[i]["start"]
                    merged_end = slides[j - 1]["end"]
                    seg_start = self.format_time(merged_start)
                    seg_end = self.format_time(merged_end)
                    print(f"  Merged segment: {count} consecutive short slides "
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

    def _reclassify_by_face(self, slides: List[Dict]) -> List[Dict]:
        """
        Post-processing pass that reclassifies 'slide' segments as 'camera' when a face
        is detected in sampled frames.

        Samples 3 frames spread across the stable portion of each slide segment and runs
        Haar Cascade face detection on each. If any sample contains a face whose area
        exceeds face_area_threshold (relative to frame area), the segment is reclassified
        as 'camera'. This catches stable full-screen camera shots that don't generate
        many rapid transitions and are therefore missed by _merge_camera_segments.

        Skips segments already classified as 'camera'. Runs in O(n_slides) time since
        Haar Cascade takes ~5 ms per frame on CPU.
        """
        cascades = [c for c in (self._face_cascade, self._profile_cascade)
                    if c is not None and not c.empty()]

        print("\n[Post-processing: camera reclassification (face + color)...]")
        for slide in slides:
            if slide["type"] == "camera":
                continue

            content_start = slide.get("content_start", slide["start"])
            stable_duration = slide["end"] - content_start
            margin = min(0.5, stable_duration * 0.15)
            t_start = content_start + margin
            t_end = slide["end"] - margin

            cap = cv2.VideoCapture(self.video_path)
            reason = None
            for k in range(3):
                t = t_start + (t_end - t_start) * k / 2 if t_end > t_start else t_start
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

                # --- color saturation heuristic ---
                # Fullscreen camera has skin tones + natural background across most of
                # the frame. PiP camera in a corner only raises saturation in a small
                # region (~15-20 % of frame), so the pixel ratio stays well below 0.40.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                sat = hsv[:, :, 1]
                if float(np.count_nonzero(sat > 60) / sat.size) > 0.40:
                    reason = "color saturation"
                    break

                # --- Haar Cascade face detection (frontal + profile) ---
                if cascades:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_eq = cv2.equalizeHist(gray)
                    h, w = frame.shape[:2]
                    frame_area = h * w
                    for cascade in cascades:
                        for img in (gray_eq, gray):
                            faces = cascade.detectMultiScale(
                                img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                            )
                            for (_, _, fw, fh) in faces:
                                if (fw * fh) / frame_area >= self.face_area_threshold:
                                    reason = "face detected"
                                    break
                            if reason:
                                break
                        if reason:
                            break
                if reason:
                    break
            cap.release()

            if reason:
                print(f"  Slide {slide['id']} [{self.format_time(slide['start'])} – "
                      f"{self.format_time(slide['end'])}] → {reason} → camera")
                slide["type"] = "camera"

        return slides

    def _merge_consecutive_noncontent(self, slides: List[Dict]) -> List[Dict]:
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
                print(f"  Merging {j - i} consecutive {seg_type} segments "
                      f"[{self.format_time(merged_start)} – {self.format_time(merged_end)}]")
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

    def _has_demo_overlay(self, frame_bgr: np.ndarray) -> bool:
        """
        Returns True when a frame appears to show an application window (terminal/IDE)
        overlaid on a light presentation slide.

        Heuristic: the frame must contain both a significant very-dark area (the
        overlay, brightness < 60) and a significant very-light area (the slide
        background, brightness > 200). Additionally the dark pixels must be
        spatially concentrated (high standard deviation across grid cells) rather
        than spread evenly like normal text on a slide.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        total = gray.size
        dark_ratio = np.count_nonzero(gray < 60) / total
        light_ratio = np.count_nonzero(gray > 200) / total

        if dark_ratio < 0.04 or light_ratio < 0.25:
            return False

        grid = 4
        h, w = gray.shape
        cell_h, cell_w = h // grid, w // grid
        dark_block_ratios = np.array([
            np.count_nonzero(gray[r * cell_h:(r + 1) * cell_h,
                                  c * cell_w:(c + 1) * cell_w] < 60) / (cell_h * cell_w)
            for r in range(grid) for c in range(grid)
        ], dtype=float)
        return float(np.std(dark_block_ratios)) > 0.12

    def _reclassify_demo_slides(self, slides: List[Dict]) -> List[Dict]:
        """
        Reclassifies 'slide' segments as 'demo' when the representative frame
        contains a dark application-window overlay on a light slide background.
        Runs after face-detection reclassification so camera segments are skipped.
        """
        print("\n[Post-processing: demo-overlay detection...]")
        for slide in slides:
            if slide["type"] != "slide":
                continue
            content_start = slide.get("content_start", slide["start"])
            mid_time = (content_start + slide["end"]) / 2
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()
            cap.release()
            if ret and self._has_demo_overlay(frame):
                print(f"  Slide {slide['id']} [{self.format_time(slide['start'])} – "
                      f"{self.format_time(slide['end'])}] → demo overlay → reclassified as demo")
                slide["type"] = "demo"
        return slides

    def _merge_similar_adjacent(self, slides: List[Dict]) -> List[Dict]:
        """
        Final deduplication pass: merges adjacent 'slide' pairs that are visually
        identical (change < similarity_threshold). Handles false transitions caused
        by minor on-screen events (cursor, tooltip, brief annotation).
        Repeats until no further merges occur.
        """
        if not slides:
            return slides

        print("\n[Post-processing: merging visually identical adjacent slides...]")
        merged = True
        while merged:
            merged = False
            new_slides: List[Dict] = []
            i = 0
            while i < len(slides):
                if (i + 1 < len(slides)
                        and slides[i]["type"] == "slide"
                        and slides[i + 1]["type"] == "slide"):
                    f1 = self._load_frame_at_time(
                        (slides[i].get("content_start", slides[i]["start"]) + slides[i]["end"]) / 2
                    )
                    f2 = self._load_frame_at_time(
                        (slides[i + 1].get("content_start", slides[i + 1]["start"]) + slides[i + 1]["end"]) / 2
                    )
                    if (f1 is not None and f2 is not None
                            and self._calculate_change_percentage(f1, f2) < self.similarity_threshold):
                        print(f"  Slides {slides[i]['id']} and {slides[i + 1]['id']} "
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

    def _best_frame_at_slide(self, slide: Dict) -> Optional[np.ndarray]:
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

        cap = cv2.VideoCapture(self.video_path)
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

    def _export_slide_images(self, slides: List[Dict]):
        """Saves the sharpest representative frame for each final (post-merge) slide."""
        if not self.output_dir:
            return
        print("\n[Exporting slide images...]")
        for slide in slides:
            frame = self._best_frame_at_slide(slide)
            if frame is not None:
                filename = f"slide_{slide['id']:03d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame)
                slide["image"] = filename
                print(f"  {filename}  [{self.format_time(slide['start'])} – {self.format_time(slide['end'])}]"
                      f"  type={slide['type']}")

    def _export_json(self, slides: List[Dict]):
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
            }
            for s in slides
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[Exported metadata → {path}]")

    def run(self) -> List[Dict]:
        """
        Runs the slide detection process.

        Returns:
            List[Dict]: A list of detected slides with start/end times.
        """
        if not self._initialize_capture():
            return []

        step_frames = int(self._fps * self.check_interval)
        if step_frames < 1:
            step_frames = 1

        current_slide_start = 0.0
        slide_idx = 1
        self.slides = []

        # Read first frame
        ret, frame = self._cap.read()
        if not ret:
            return []

        prev_frame_processed = self._preprocess_frame(frame)
        frame_counter = 0

        print(f"\n[Processing...]")

        n_checks = self._total_frames // step_frames
        pbar = tqdm(total=n_checks, desc="Detected: 0", unit="checks")

        while True:
            # Skip step_frames-1 frames using grab() — no full decode needed
            for _ in range(step_frames - 1):
                if not self._cap.grab():
                    break
            ret, current_frame = self._cap.read()
            if not ret:
                break

            frame_counter += step_frames
            current_time = frame_counter / self._fps
            pbar.update(1)

            current_frame_processed = self._preprocess_frame(current_frame)
            change_pct = self._calculate_change_percentage(prev_frame_processed, current_frame_processed)

            changed_blocks = (
                self._count_changed_blocks(prev_frame_processed, current_frame_processed)
                if change_pct > self.threshold_percent else 0
            )

            if change_pct > self.threshold_percent and changed_blocks >= self.min_changed_blocks:
                if self.confirm_transitions and not self._confirm_transition(prev_frame_processed, current_time):
                    # Scene reverted one interval later → single-frame glitch, ignore
                    prev_frame_processed = current_frame_processed
                else:
                    duration = current_time - current_slide_start

                    # Record slide regardless of duration – post-processing handles short ones
                    self.slides.append({
                        "id": slide_idx,
                        "start": current_slide_start,
                        "end": current_time,
                        "duration": duration,
                        "type": "slide",
                        "image": None,
                        "content_start": current_slide_start,
                    })

                    slide_idx += 1
                    current_slide_start = current_time
                    prev_frame_processed = current_frame_processed
                    pbar.set_description(f"Detected: {slide_idx - 1}")

            else:
                prev_frame_processed = current_frame_processed

        pbar.close()

        # Final slide
        final_duration = self._duration - current_slide_start
        self.slides.append({
            "id": slide_idx,
            "start": current_slide_start,
            "end": self._duration,
            "duration": final_duration,
            "type": "slide",
            "image": None,
            "content_start": current_slide_start,
        })

        self._cap.release()

        # Post-process: merge clusters of short slides (camera feed or fast demo) into one slide.
        # Must run BEFORE _merge_short_slides – once that step absorbs short slides into longer
        # neighbours they exceed min_duration and the camera detection misses the whole run.
        self.slides = self._merge_camera_segments(self.slides)

        # Post-process: merge remaining isolated short slides that match a neighbour
        self.slides = self._merge_short_slides(self.slides)

        # Post-process: reclassify stable camera segments missed by duration-based detection
        if self.use_face_detection:
            self.slides = self._reclassify_by_face(self.slides)

        # Post-process: reclassify slides that show a demo/IDE overlay TODO not good enough yet, needs improvement
        # self.slides = self._reclassify_demo_slides(self.slides)

        # Post-process: merge consecutive segments of the same non-content type (camera / demo)
        self.slides = self._merge_consecutive_noncontent(self.slides)

        # Post-process: merge visually identical adjacent slides (false transitions)
        self.slides = self._merge_similar_adjacent(self.slides)

        # Export final slide images (after merging, with correct numbering)
        self._export_slide_images(self.slides)

        # Export JSON metadata for downstream pipeline consumption
        self._export_json(self.slides)

        print("\n" + "=" * 60)
        for slide in self.slides:
            print(f"Slide {slide['id']} [{slide['type']:6s}]: {self.format_time(slide['start'])} - "
                  f"{self.format_time(slide['end'])} ({slide['duration']:.2f}s)")
        print("=" * 60 + "\n")

        return self.slides


if __name__ == "__main__":
    video_files =["videos/test_video_1.mp4", "videos/test_video_2.mp4", "videos/test_video_3.mp4",
                  "videos/test_video_4.mp4", "videos/test_video_5.mp4"]
    # video_files = ["videos/test_video_3.mp4"]
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    for video_file in video_files:
        detector = SlideDetector(
            video_file,
            output_dir="detected_slides",
            threshold_percent=5,
            min_duration=10,
            similarity_threshold=5,
            min_changed_blocks=4,       # ignore changes in < 4/16 blocks (e.g. PiP camera corner)
            camera_segment_min_count=5, # remove runs of >=5 consecutive short slides (fullscreen camera)
            face_area_threshold=0.15,
        )
        slides = detector.run()