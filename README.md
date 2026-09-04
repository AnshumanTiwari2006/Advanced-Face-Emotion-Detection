# Real-Time Facial Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)]()

A robust, real-time computer vision application designed to detect faces and classify human emotions using a Convolutional Neural Network (CNN).

The system captures live webcam feed, isolates faces using OpenCV's Haar cascades, and runs a forward pass through a pre-trained Keras model to classify the subject's expression into one of seven fundamental emotional states. To ensure reliability in continuous frame sequences, the application implements temporal smoothing techniques to prevent erratic classifications and features comprehensive CSV telemetry logging for downstream analysis.

---

## Technical Architecture & Key Features

### 1. Computer Vision & Inference Pipeline
* **Face Detection:** Utilizes OpenCV's built-in `haarcascade_frontalface_default.xml` to detect multiple face bounding boxes per frame efficiently.
* **Pre-processing:** Extracts the facial Region of Interest (ROI), converts it to grayscale, and resizes it to a normalized `(64, 64)` tensor. The pixel intensities are scaled to `[0, 1]` before being passed to the model.
* **Deep Learning Model:** A pre-trained TensorFlow/Keras model (`emotion_model.h5`) processes the `(1, 64, 64, 1)` tensor to output a probability distribution across 7 classes: `Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral`.

### 2. Temporal Smoothing Matrix
Frame-by-frame emotion classification is inherently noisy due to micro-expressions and lighting variations. This system implements a **Rolling Average Buffer** using `collections.deque`. By maintaining a history of the last $N$ predictions (default: 10 frames), the system calculates the mean probability distribution across the temporal window. This significantly reduces flickering and provides a highly stable, continuous classification output.

### 3. Real-time Diagnostic HUD
The application features a custom diagnostic visualization panel rendered directly alongside the video feed using NumPy array manipulation (`np.hstack`). It provides:
* **Bounding Box Tracking:** Dynamically colored bounding boxes mapped to the dominant emotion.
* **Probability Bar Chart:** A live, updating vertical bar chart visualizing the confidence distribution across all 7 classes.
* **Performance Metrics:** Real-time FPS (Frames Per Second) and face count tracking.

### 4. Telemetry & Data Logging
For analytical purposes and dataset generation, the system logs the session data asynchronously. Every 30 frames, a record is appended to `emotion_log.csv` containing:
* Precise timestamp.
* The dominant emotion class.
* The peak confidence score.
* The exact probability distribution across all 7 emotion classes.

## Dataset & Pipeline Alignment

The model's foundation is built upon the standard **FER2013 Dataset**. To ensure the live webcam feed exactly matches the conditions the neural network was trained on, our real-time ingestion pipeline explicitly mirrors the dataset's constraints.

**Dataset Specifications:**
* **Total Images:** 35,887 facial images
* **Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

**Pipeline Proof (Live Feed → FER2013 Format):**
When a face is detected via Haar Cascades, the system does not feed the raw colored image to the model. Instead, the ROI undergoes strict mathematical preprocessing to match the FER2013 training distribution:
1. **Grayscale Conversion:** `cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)` strips the RGB channels, matching the dataset's single-channel format.
2. **Dimensional Normalization:** The bounding box is resized strictly to `(64, 64)` (an upscaled variant of the original 48x48 architecture for enhanced spatial feature extraction).
3. **Tensor Scaling:** The pixel array is cast to `float32` and normalized via `/ 255.0`, ensuring the CNN weights compute against `[0, 1]` vectors rather than `[0, 255]`.

---

## Results

* **Validation Accuracy:** **~66%** *(This is the recognized benchmark ceiling for lightweight, single-model CNN architectures on the challenging FER2013 dataset in the wild).*
* **Memory Footprint:** The model weights (`emotion_model.h5`) are extremely optimized at just **872 KB**. This allows the entire inference graph to run purely on the CPU without bottlenecking the main execution thread.
* **Inference Speed:** Fluid Real-Time Webcam Inference (<50ms latency per frame).
* **Capability:** Resolves 7 Distinct Emotion Classes deterministically.
* Try it out instantly and make sure it is fully loaded. 

---

## Setup and Installation

### Prerequisites
* Python 3.8 or higher (Tested on Python 3.12)
* A working webcam.

### 1. Environment Setup
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### 2. Install Dependencies
Install the required machine learning and computer vision libraries:
```bash
pip install tensorflow opencv-python numpy
```

### 3. Model Assets
Ensure that your pre-trained model weights file (`emotion_model.h5`) is located in the root directory of the project. The Haar cascade XML file is fetched automatically from your local OpenCV library installation.

---

## Execution

To start the emotion detection application, run the core script from your terminal:

```bash
python new.py
```

### Controls
* **Quit Application:** Press the `q` key while focused on the video window to safely terminate the session, release the camera hardware, and save the telemetry log.

---

## 📊 Data Output Format
Upon running the application, a file named `emotion_log.csv` will be generated or appended to in the root directory.

**Sample Output:**
```csv
timestamp,emotion,confidence,Anger,Disgust,Fear,Happy,Sad,Surprise,Neutral
2026-06-12 21:40:01,Happy,0.8521,0.0012,0.0101,0.0210,0.8521,0.0115,0.0020,0.1021
2026-06-12 21:40:02,Happy,0.8644,0.0010,0.0090,0.0190,0.8644,0.0100,0.0020,0.0946
```
This structured data allows for seamless ingestion into analytical pipelines like Pandas or data visualization software to track emotional sentiment over time.
