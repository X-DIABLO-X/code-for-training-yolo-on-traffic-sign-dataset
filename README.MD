🚦 YOLOv8 Traffic Sign Detection
This project uses YOLOv8 to train an object detection model that can identify traffic signs from images. The model is trained on a dataset of 20,000+ labeled images and is capable of detecting different classes of traffic signs in real-time.

📁 Dataset
Dataset Source: /kaggle/input/traffic-sign-detection-dataset/Traffic_sign_detection_data/

Structure:

bash
Copy
Edit
Traffic_sign_detection_data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
Images are in JPG/PNG format.

Labels are in YOLO format (.txt with normalized [x_center, y_center, width, height]).

⚙️ Environment
Make sure the following libraries are installed:

bash
Copy
Edit
pip install ultralytics opencv-python matplotlib
✅ YOLOv8 comes from the ultralytics library.

🏋️ Training
Step 1: Check for GPU
python
Copy
Edit
device = "cuda" if torch.cuda.is_available() else "cpu"
Step 2: Train the YOLOv8 model
python
Copy
Edit
from ultralytics import YOLO

# Load pretrained YOLOv8 model (medium)
model = YOLO("yolov8m.pt")

# Train on traffic sign dataset
model.train(
    data="/kaggle/working/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=device,
    workers=4,
    project="runs/traffic_sign_detection",
    name="yolo_v8m_traffic_signs",
    exist_ok=True
)
Optional Fine-Tuning:
python
Copy
Edit
model.train(
    data="/kaggle/working/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=device,
    workers=4,
    project="runs/traffic_sign_detection",
    name="yolo_v8m_traffic_signs_finetune",
    resume=True
)
🧪 Inference and Visualization
Load and Run Predictions on Test Images
python
Copy
Edit
from ultralytics import YOLO
import cv2, os
import matplotlib.pyplot as plt

model = YOLO("runs/traffic_sign_detection/yolo_v8m_traffic_signs/weights/best.pt")

test_dir = "/kaggle/input/traffic-sign-detection-dataset/Traffic_sign_detection_data/images/test"
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

for img_path in image_files[:10]:
    results = model.predict(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
📦 Model Export
After training:

python
Copy
Edit
model.export(format="torchscript")
The model will be saved as:

bash
Copy
Edit
runs/traffic_sign_detection/yolo_v8m_traffic_signs/weights/best.torchscript
📊 Evaluation Metrics
Use the built-in validation:

bash
Copy
Edit
yolo val task=detect model=best.pt data=/kaggle/working/data.yaml imgsz=640
You’ll get:

mAP@0.5

Precision

Recall

✅ Results
Trained on ~20k images

Achieved real-time detection (~6ms per image on GPU)

Exportable to TorchScript, ONNX, OpenVINO
