import numpy as np 
import cv2 
from Algorithm_1 import alg_1

def pose_storer (video_path):
    cap = cv2.VideoCapture(video_path)
    frame_to_pose_dict = {}
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        #average_face_mesh = person_talking_detector(frame)
        if ret == False:
            break
        accurate_pose = alg_1(frame, avg_face_mesh)
        frame_to_pose_dict[i] = accurate_pose
    
    def temporal_coherence (frame_to_pose_dict):
    
    
    
    return frame_to_pose_dict





        


    
    

