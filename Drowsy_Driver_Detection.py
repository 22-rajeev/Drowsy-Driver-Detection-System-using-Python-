import cv2                                    # For capturing video frames from cameras or video files.
import imutils                                # Imutils is a set of convenience functions to make basic image processing functions such as resizing, rotating, and displaying images easier with OpenCV.
from imutils import face_utils                # Face utils is a collection of functions specifically designed for facial analysis tasks.
import dlib                                   # This Library is for facial recognition.
from scipy.spatial import distance            # For calculation of distance Facial landmark detection .
from pygame import mixer                      # Library for adding music.

mixer.init()                                  # Initializing mixer. 
mixer.music.load("Models/Warning Alarm.wav")  # Setting path and loading music for the warning sound.

face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # face_cap is the file for complete face detection.
detect = dlib.get_frontal_face_detector()                                                        # dlib's inbuilt frontal face detector function.
predict = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")                   # .xml file that divides the face into 68 facial detection points.




(LStart , LEnd)= face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]            # Variables for Left eye detection.
(RStart , REnd)= face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]           # Variables for Right eye detection.



def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1],eye[5])     # The Human is eye recognized through 6 (Starting from 0 - 6) points on the eye. 
    B = distance.euclidean(eye[2],eye[4])     # A = First Vertical Distance , B = Second Vertical distance.
    C = distance.euclidean(eye[0],eye[3])     # Horizontal distance.
    ear = (A + B)/(2.0 * C)                   # Calculate the eye aspect ratio when eye objects are passed.
    return ear                                # Returns EAR for a single eye.




thresh = 0.25                   # Eye aspect ratio should not go less than this threshold value. 
flag =0                         # Initialzing flag  variable to 0 for checking if EAR  == 0 for time grater than frame checker.
frame_checker = 25              # If EAR value goes less than threashold value for 20 frames than Warning is generated.  



capture =cv2.VideoCapture(0)                       # Object for video Capture.
while True:                                        # Infinite Loop for Turning on the Camera.
    ret , frame = capture.read()                   # This line reads a frame from a video source, storing the success status in ret and the frame data in frame.
    col = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converts the Background color to black and white for facial muscle detection.
    faces = face_cap.detectMultiScale(             # This is a pre-trained cascade classifier.
        col,                                       # A cascade classifier is a machine learning-based object detection algorithm that is used to detect objects in images or video streams.
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h)in faces:                                  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)     # This for loop generates a Green Box around the face using cv2.rectangle() function. 
    


    subjects = detect(col , 0)                      # Detect faces in the grayscale image using the frontal face detector.
    for subject in subjects:                        # Iterate through each detected face.
        shape = predict(col , subject)              # Predict the facial landmarks for the current face.
        shape = face_utils.shape_to_np(shape)       # Convert the predicted landmarks to NumPy array of x,y coordinates.
        lefteye = shape[LStart:LEnd]                # Extracting the coordinates of the left eye from the facial landmarks.
        righteye = shape[RStart:REnd]               # Extract the coordinates of the right eye from the facial landmarks.


        leftEAR  = eye_aspect_ratio(lefteye)        # Calling eye_aspect_ratio function() for left eye.
        rightEAR = eye_aspect_ratio(righteye)       # Calling eye_aspect_ratio function() for right eye.
        EAR = (leftEAR+rightEAR)/2.0                # Effective eye aspect ratio of bothe the eyes will give us a test for drowsiness.


        LefteyeHULL = cv2.convexHull(lefteye)                       # Draws the Convex Hull for left eye. The convex hull is the smallest convex shape that encloses all given points.
        RighteyeHULL = cv2.convexHull(righteye)                     # Draws the Convex Hull for right eye.
        cv2.drawContours(frame,[LefteyeHULL],-1,(0,255,0),1)        # Generates Contours for left eye.
        cv2.drawContours(frame,[RighteyeHULL ],-1,(0,255,0),1)      # Generates Contours for right eye.
       
       
       
        if EAR < thresh:
            flag += 1                                                                                           # Increments the value of flag for which EAR < Threshold EAR. 
            print(flag)
            if flag >= frame_checker:
                cv2.putText(frame, "Warning!", (440, 680), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5, (0, 0, 255), 5)  # Generates a text on the frame as Warning! if drowsiness is detected.
                mixer.music.play()                                                                              # Plays Warning music.
        else:
            flag = 0         #Sets the value of flag again to 0.



    cv2.imshow("Drowsy Driver", frame)  # imshow function creates a frame for the image to be displayed takin input as frame heading + video capture object.
    if cv2.waitKey(10) == ord("q"):     # If q is pressed the infinite loop breaks.
        break

    
capture.release()           # Releases the video capture object
cv2.destroyAllWindows()     ## Close all OpenCV windows that are currently open, freeing up system resources.T