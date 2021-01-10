import numpy as np
import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-f", "--face", required=True, help="Path to face detector model directory")
ap.add_argument("-a", "--age", required=True, help="Path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")

args = vars(ap.parse_args)

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]

# Load pretrained Face detector model
print("[INFO] Loading face detector model...")
face_prototxt_path = os.path.join([args["face"], "deploy.prototxt"])
face_weights_path = os.path.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(face_prototxt_path, face_weights_path)

# Load pretrained age detection model
print("[INFO] Loading age detector model...")
age_prototxt_path = os.path.join([args["age"], "age_deploy.prototxt"])
age_weights_path = os.path.join([args["age"], "age_net.caffemodel"])
age_net = cv2.dnn.readNet(age_prototxt_path, age_weights_path)

