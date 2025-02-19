# FaceRecognitionKNN

## Overview

FaceRecognitionKNN is a machine learning-based face recognition system using the **K-Nearest Neighbors (KNN) algorithm**. It utilizes **OpenCV** for face detection and **NumPy** for efficient data processing, allowing real-time facial recognition.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade to detect faces in images or video streams.
- **Face Data Collection**: Captures and stores face data in `.npy` format.
- **Feature Extraction**: Converts images to structured numerical arrays for recognition.
- **Face Classification with KNN**: Uses the K-Nearest Neighbors algorithm for identifying faces.
- **Real-time Face Recognition**: Identifies known faces in live video.
- **Demo Scripts**: Includes sample scripts for testing face detection and recognition.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- OpenCV
- NumPy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/FaceRecognitionKNN.git
   cd FaceRecognitionKNN
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
3. Run the face data collection script to store face embeddings:
   ```bash
   python face-detection-app/faceDataCollection.py
   ```
4. Train the model and start face recognition:
   ```bash
   python face-detection-app/faceRecognize.py
   ```

## File Structure

```
FaceRecognitionKNN/
│── Demo/
│   ├── frontFaceDemo.py       # Basic face detection demo
│   ├── videoFromCvDemo.py     # Real-time face detection from video
│   ├── haarcascade_frontalface_alt.xml # Haar Cascade model
│
│── face-detection-app/
│   ├── faceDataCollection.py  # Collects face data
│   ├── faceRecognize.py       # Recognizes faces in real-time
│   ├── face_recognition_knn.py # KNN-based face recognition
│   ├── image2npy.py           # Converts images to NumPy arrays
│   ├── haarcascade_frontalface_alt.xml # Haar model for detection
│
│── face-recognition-npy/
│   ├── harsha.npy             # Stored face data
│
│── README.md
```

## How It Works

### 1. Data Collection

- Captures images using OpenCV.
- Detects faces using Haar Cascade.
- Saves face data as NumPy arrays in `.npy` format.

### 2. Preprocessing

- Converts images to grayscale.
- Extracts face features and stores them in a structured format.

### 3. Face Recognition

- Loads stored `.npy` data.
- Uses the **KNN algorithm** to classify detected faces.
- Matches new faces against stored embeddings and labels them accordingly.

## Performance Enhancements

- **Optimized for Speed**: Uses NumPy for fast data processing.
- **Improved Accuracy**: Proper feature extraction using Haar Cascades.
- **Real-time Processing**: Supports video-based recognition.

## Future Improvements

- Replace Haar Cascades with **MTCNN or Dlib** for better face detection.
- Implement **deep learning-based face embeddings (FaceNet/Dlib)** instead of KNN.
- Introduce a **GUI for face registration and recognition**.

## Contribution

If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-xyz`).
5. Open a pull request.

---

Developed by **Harsha Vardhan G**🚀

