#!/usr/bin/env python
# coding: utf-8

# In[2]:


import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


process_this_frame = True


cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if process_this_frame:

        #smaller image for speed
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame,model='cnn')
        name = "Fakhir"
    
    process_this_frame = not process_this_frame #process every other imaage
    
    if len(face_locations) > 0:
        (top,right,bottom,left) = face_locations[0]
        
##        top *= 4
##        right *= 4
##        bottom *= 4
##        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #press q to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[7]:


cap.release()
cv2.destroyAllWindows()


# In[31]:





# In[ ]:




