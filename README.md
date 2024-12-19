# **Drowsy Driver Detection System Using Python**

Utilizes Python libraries to accurately detect driver drowsiness in real-time through facial expression and eye movement analysis.

---

### ** Folder Structure**

Drowsy_Driver_Detection.py  # Main Python script
Models  - Warning Alarm.wav                      # Alarm sound file for drowsiness alert
        - shape_predictor_68_face_landmarks.dat  # Pre-trained model for facial landmark detection
---
### ** Prerequisites**
Make sure you have the following libraries installed before running the program.

**Required Libraries:**
- **OpenCV**: For capturing video frames and image processing.
- **imutils**: Utility functions for image manipulation.
- **dlib**: Facial landmark detection for detecting and tracking face/eye movements.
- **scipy**: Calculates Euclidean distance for eye aspect ratio (EAR) analysis.
- **pygame**: Plays alert sounds when drowsiness is detected.
