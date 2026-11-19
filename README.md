# RoadGuard-AI---Road-Damage-Detection
# RoadGuard AI

## AI-Based Road Damage Detection System

RoadGuard AI is an AI-powered road damage detection system that uses YOLOv8 to detect road damage from road images.

The system provides a Flask web interface where users can upload a road image and receive an annotated detection result, confidence scores, severity estimation, detection history, and a downloadable PDF inspection report.

---

## Features

- AI-based road damage detection
- YOLOv8 object detection
- Road image upload
- Pothole detection
- Crack and other road damage detection
- Detection summary
- Confidence scores
- Automatic severity estimation
- Detection history
- SQLite database
- Downloadable PDF inspection report
- Flask-based web application
- Local CPU support

---

## Detected Damage Classes

The trained model contains 7 road-damage classes:

1. Alligator
2. Block
3. Crack
4. Edge
5. Longitudinal
6. Pothole
7. Transverse

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO implementation |
| Flask | Web application |
| OpenCV | Image processing |
| SQLite | Detection history database |
| HTML/CSS | Frontend |
| ReportLab | PDF report generation |
| Git/GitHub | Version control |

---

## Project Structure

```text
RoadGuard-AI/
|
|-- app.py
|-- train.py
|-- requirements.txt
|-- yolov8n.pt
|-- .gitignore
|-- README.md
|
|-- models/
|   `-- roadguard_best.pt
|
`-- templates/
    `-- index.html


Installation

1. Clone the repository

git clone https://github.com/Shivamrai11/RoadGuard-AI.git
cd RoadGuard-AI

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

Windows: venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

Run the Application

Start the Flask application: python app.py

Open the following address in your browser: http://127.0.0.1:5000

How It Works

 User
  |
  v
Upload Road Image
  |
  v
Flask Web Application
  |
  v
YOLOv8 Model
  |
  v
Road Damage Detection
  |
  +-- Damage Class
  +-- Confidence Score
  +-- Bounding Box
  +-- Estimated Area
  |
  v
Severity Estimation
  |
  v
SQLite Detection History
  |
  v
PDF Inspection Report


Detection Output

The system provides:

Damage type
Damage count
Confidence percentage
Bounding boxes
Estimated image area
Severity level
Detection history
PDF inspection report


Trained Model

The trained model is stored at: models/roadguard_best.pt

The Flask application automatically loads this model when the application starts.


Training

The training script is: train.py

The project uses YOLOv8 for road damage object detection.


Future Improvements

Real-time video detection
Mobile application
GPS-based damage mapping
Automatic maintenance alerts
Cloud deployment
Improved severity estimation
Larger and more diverse training datasets


Author

Shivam Rai

B.Tech Computer Science and Engineering



License

This project is intended for educational and academic purposes.s
