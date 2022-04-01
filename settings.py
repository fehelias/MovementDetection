import cv2

INPUT_VIDEO = cv2.VideoCapture(r"C:\Users\fehel\Desktop\detect_objects\media\train_objects.mp4")

CARS_DETECTION  = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=200)
TRAIN_DETECTION1 = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=20)
TRAIN_DETECTION2 = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=20)
TRAIN_DETECTION3 = cv2.createBackgroundSubtractorMOG2(history=8,varThreshold=20)
