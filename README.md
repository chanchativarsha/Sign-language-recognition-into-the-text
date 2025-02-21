# Sign-language-recognition-into-the-text
Sign language is primarily used by the deaf and mute, making communication with non-signers challenging. To bridge this gap, a Sign Language Recognition (SLR) system is designed to enhance communication. It captures sign expressions from a hearing-impaired person and converts them into text or voice for easy understanding by others. 

Below figure shows an example from every class of sign images dataset.

![image](https://github.com/user-attachments/assets/debdcef8-2aaf-41ae-9f32-0632d34e8e98)

Technologies Used
1.1 Mediapipe
Purpose: Hand detection and keypoint extraction.
How it works:
Hand Landmarks: Mediapipe uses a machine learning pipeline to detect 21 keypoints on a hand in 3D space.
Tracking: Tracks the position of the hands frame by frame, ensuring stability even with partial occlusions.
Key Features:
Real-time performance.
Robust to varying lighting and background conditions.
Efficient, designed for mobile and desktop platforms.
1.2 OpenCV
Purpose: Video capture and visualization.
How it works:
Captures the webcam feed in real time.
Processes and crops frames to focus on the region of interest.
Displays visual outputs such as bounding boxes, text overlays, and prediction bars.
1.3 TensorFlow/Keras
Purpose: Machine learning model implementation for sequence prediction.
How it works:
Model Type: Long Short-Term Memory (LSTM), a recurrent neural network architecture.
Processes sequences of 30 frames to classify gestures based on temporal dynamics of hand movements.
1.4 NumPy
Purpose: Data manipulation.
How it works:
Manages sequences of extracted keypoints for LSTM input.
Performs numerical operations like array slicing and data preprocessing.
2. Algorithms Used
2.1 Mediapipe's Hand Detection
Technology: Uses Convolutional Neural Networks (CNNs).
Pipeline:
Palm Detection: Identifies hand location using a lightweight neural network.
Landmark Localization: Detects 21 3D hand keypoints.
The system combines both tasks in a computationally efficient manner for real-time tracking.
2.2 Keypoint Extraction
Mediapipe outputs 21 3D keypoints ([x, y, z]) for each hand. These are flattened into a single feature vector:
python
Copy code
rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
2.3 LSTM-based Gesture Classification
Algorithm: Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN).
Why LSTM:
Designed for sequential data, such as time-series or video frames.
Captures temporal dependencies, crucial for understanding gesture dynamics.
Input to LSTM:
A sequence of 30 frames, each containing 63 keypoints (21 landmarks × 3 coordinates).
Data shape: (batch_size, sequence_length, feature_size) = (1, 30, 63).
2.4 Action Smoothing
Algorithm: Mode Filtering (Custom Implementation).
Why: Reduces noise and ensures consistent predictions.
How:
Looks at the last 10 predictions and ensures consistency:
python
Copy code
if np.unique(predictions[-10:])[0] == np.argmax(res):
3. Workflow Overview
Hand Detection and Keypoint Extraction:
Mediapipe detects hands and extracts 3D keypoints.
Keypoint Processing:
Keypoints are flattened and stored in a sliding window of the last 30 frames.
Gesture Prediction:
The LSTM model processes the sequence to classify gestures.
Output Visualization:
OpenCV overlays the predictions and confidence scores on the video feed.
4. Why This Technology Stack?
Real-Time Performance:

Mediapipe is highly optimized for real-time applications.
OpenCV handles video capture and processing efficiently.
Temporal Modeling:

LSTM excels at capturing sequential patterns in time-series data.
Scalability:

The system can be extended to recognize more gestures by retraining the LSTM with new data.
Ease of Use:

TensorFlow/Keras and Mediapipe offer high-level APIs, simplifying implementation.
5. Applications of Algorithms
Gesture Recognition:
Combines spatial features (Mediapipe) and temporal features (LSTM).
Sign Language Recognition:
Detects static signs and gestures with temporal motion.
Human-Computer Interaction:
Enables gesture-based control in games, AR/VR, or IoT devices.
This combination of technologies and algorithms allows for robust, efficient, and scalable gesture recognition.

output:
![image](https://github.com/user-attachments/assets/56a1ac54-126b-482f-9216-36040bd2f568)
