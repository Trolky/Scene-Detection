"""Pure frame-processing functions — stateless, no I/O.

Shared across the whole pipeline: detection loop and post-processing passes.
"""
import cv2
import numpy as np


def format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Converts frame to grayscale and applies blurring to reduce noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def change_percentage(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculates the percentage of changed pixels between two frames."""
    frame_delta = cv2.absdiff(frame1, frame2)
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    return (changed_pixels / total_pixels) * 100


def count_changed_blocks(frame1: np.ndarray, frame2: np.ndarray, grid: int = 4) -> int:
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