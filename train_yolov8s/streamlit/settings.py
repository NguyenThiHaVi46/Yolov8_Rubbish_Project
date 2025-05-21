from pathlib import Path
import sys


file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default_img.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detected_img.jpg'

# ML Model config
MODEL_DIR =  ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR  / 'yolov8.pkl'
YOLO_PT_DIR = MODEL_DIR / 'yolo_base.pt'

# config Confidence
DEFAULT_CONFIDENCE = 0.6
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
