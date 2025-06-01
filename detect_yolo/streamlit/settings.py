from pathlib import Path
import sys

# Lấy đường dẫn tuyệt đối của tệp hiện tại
file_path = Path(__file__).resolve()

# Lấy thư mục cha của tập tin hiện tại
root_path = file_path.parent

# Thêm đường dẫn gốc vào danh sách sys.path nếu nó chưa có ở đó
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Lấy đường dẫn tương đối của thư mục gốc liên quan đến thư mục làm việc hiện tại
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

# Sources list: hiện tại chỉ có loại image
SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default_img.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detected_img.jpg'

# ML Model config
MODEL_DIR =  ROOT / 'weights'
LAST_MODEL = MODEL_DIR  / 'last.pt'
BEST_MODEL = MODEL_DIR / 'best.pt'

# config Confidence
DEFAULT_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 1.0
