#!/usr/bin/env python
# coding: utf-8

# In[2]:


import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[43]:


names_file = open("Names.txt",'a')


# In[44]:


name = input("what is your name? ")


# In[45]:


names_file.write(name + "\n")


# In[46]:


names_file.close()


# In[48]:


os.mkdir(f"Data/{name}")


# In[52]:


process_this_frame = True
idx = 0
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    save_fr = frame.copy()
    if process_this_frame:
        idx += 1
        #smaller image for speed
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame
        

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame,model='cnn')
        
        #face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame,model='small')
        
    
    process_this_frame = not process_this_frame #process every other imaage
    
    if len(face_locations) > 0:
        cv2.imwrite(f"Data/{name}/{name}_{idx}.jpg",save_fr)
        (top,right,bottom,left) = face_locations[0]
        
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4
        
#         imgs.append(save_fr[top:bottom , left:right])
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#         for face_landmarks in face_landmarks_list:
#             for marks in face_landmarks.keys():
#                 for point in face_landmarks[marks]:
#                     cv2.circle(frame,point,2,[0,255,0])
        
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #press q to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




