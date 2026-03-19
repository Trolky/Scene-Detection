from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import sys

def find_scenes(video_path, threshold=27.0):
    # Open the video using the new API (replaces VideoManager)
    try:
        video = open_video(video_path)
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    scene_manager = SceneManager()

    # ContentDetector compares the difference between frames.
    # threshold determines how much change is needed to trigger a scene cut.
    # Lower value = more sensitive (detects smaller changes).
    # Default 30.0 might be too high for slides with small text changes.
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))

    # Perform scene detection on video object.
    print(f"Analyzing {video_path} with threshold {threshold}...")
    scene_manager.detect_scenes(video=video, show_progress=True)

    # Each returned scene is a tuple of the (start, end) timecode.
    scene_list = scene_manager.get_scene_list()

    print(f'Found {len(scene_list)} slides.')

    if len(scene_list) == 0:
        print("Tip: If 0 slides were found, try lowering the threshold (e.g. to 10 or 15).")
        print("Usage: python scenedetect_example.py video.mp4 [threshold]")

    for i, scene in enumerate(scene_list):
        start = scene[0].get_timecode()
        end = scene[1].get_timecode()
        duration = scene[1].get_seconds() - scene[0].get_seconds()
        print(f'Slide {i+1}: {start} - {end} (Duration: {duration:.2f}s)')

if __name__ == "__main__":

    video_file = "test_video.mp4"
    threshold = 2.0

    if len(sys.argv) > 1:
        video_file = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print("Invalid threshold value, using default.")

    find_scenes(video_file, threshold)
