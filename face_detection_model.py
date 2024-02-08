#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install opencv-python')


# In[2]:


import cv2


# In[15]:


# Load the Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the video capture
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_color = (0, 255, 0)  # Green for a face by default

        # Extract the region of interest (ROI) for eyes
        roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            if ew < 10 or eh < 10:
                face_color = (255, 0, 0)  # Red for closed eyes
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # Green for open eyes

        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)  # Set the face rectangle color

    num_faces = len(faces)

    cv2.putText(frame, f'Faces Detected: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Video', frame)

    if cv2.waitKey(10) == ord('a'):
        break

video_cap.release()
cv2.destroyAllWindows()







