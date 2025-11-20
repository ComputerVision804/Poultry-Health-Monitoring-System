# Poultry-Health-Monitoring-System
A real-time computer vision system designed to help poultry farmers monitor flock health through intelligent behavior analysis, movement tracking, and early illness detection.

🚀 Overview
This project analyzes live poultry footage to detect abnormal or inactive behavior that may indicate early signs of illness. By combining deep learning, segmentation, optical flow, and a context-aware behavior model, the system provides high-accuracy, real-time health insights.

🔧 Features
✔️ Real-Time Computer Vision Pipeline
YOLOv11 – high-speed chicken detection
SAM2 – precise chicken segmentation
ByteTrack – consistent ID tracking across frames
Optical Flow – fine-grained micro-movement analysis

✔️ Intelligent Behavioral Modeling
The system classifies activity using:
Body speed
Head movement
Direction-based optical flow
Segmentation stability
This ensures correct differentiation between resting, feeding, walking, and inactivity.

✔️ Live Four-Panel Dashboard
FPS, latency, and system performance
Per-chicken inactivity ranking
Behavior distribution charts
Flock-level health score
Real-time alerts for abnormal behavior

🎯 Why This Matters
This AI system provides:
Early illness detection
24/7 automated monitoring
Reduced mortality
Improved welfare and farm efficiency
Data-driven decision making for farmers

🧠 Tech Stack
Python
PyTorch
OpenCV
Ultralytics YOLO
SAM2
Supervision
NumPy

📁 Project Structure
├── src/
│   ├── detection/      # YOLOv11 model & inference
│   ├── segmentation/   # SAM2 segmentation pipeline
│   ├── tracking/       # ByteTrack integration
│   ├── behavior/       # Movement + optical flow analysis
│   ├── dashboard/      # Live UI & visualizations
│   └── utils/
├── models/
├── configs/
├── requirements.txt
└── README.md

▶️ How to Run
git clone <repo-url>
cd poultry-health-monitoring
pip install -r requirements.txt
python main.py --source your_video.mp4

📌 Future Improvements
Thermal camera integration
Disease prediction model
Cloud dashboard + alerts

