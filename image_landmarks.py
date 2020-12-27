import face_alignment
import cv2
import numpy as np
import torch

class image_landmarks:

        def __init__(self, left_eye, right_eye, nose_point, all_points):
                self.all_points = all_points
                self.predictions = predictions
                self.left_eye = left_eye
                self.right_eye = right_eye
                self.nose_point = nose_point

        def predictions(self):
                fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
                preds = fa.get_landmarks_from_image(self)
                preds = preds[0]
                return preds

        def all_points(self):
                all_points = []
                for i in range (len(self.predictions())):
                        x = self.predictions()[i, 0]
                        y = self.predictions()[i, 1]
                        all_points.append((x, y))
                return all_points



        def left_eye (self):
                x, y = self.predictions()[36:42, 0], self.predictions()[36:42, 1]
                x_avg, y_avg = sum(x)/len(x), sum(y)/len(y)
                return x_avg, y_avg

        def right_eye (self):
                x, y = self.predictions()[42:48, 0], self.predictions()[36:42, 1]
                x_avg, y_avg = sum(x)/len(x), sum(y)/len(y)
                return x_avg, y_avg

        def nose_point (self):
                x, y = self.predictions()[30:31, 0], self.predictions()[36:42, 1]
                x_avg, y_avg = sum(x)/len(x), sum(y)/len(y)
                return x_avg, y_avg


