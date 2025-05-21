import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("E:/Workspace/Đồ án/Đồ án 2/Waste-Classification-using-YOLOv8/streamlit-detection-tracking - app/weights/best.pt")
image_path = "E:/Rubbish_Project/Rubbish_Project/train_yolov8s/test_images/img_002.png"

# Predict
results = model.predict(source=image_path, conf=0.2, iou =0.6, save=True)

# In kết quả
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        xyxy = box.xyxy[0].tolist()
        print(f"class:{cls}, confidence: {conf:.2f}, BBox: {xyxy}")

# Lấy thư mục YOLO đã lưu kết quả
save_dir = results[0].save_dir

# Tìm file ảnh trong thư mục đã lưu
for file_name in os.listdir(save_dir):
    if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
        result_image_path = os.path.join(save_dir, file_name)
        break
else:
    raise FileNotFoundError("Không tìm thấy ảnh kết quả trong thư mục lưu.")

# Đọc và hiển thị ảnh
image = cv2.imread(result_image_path)
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
