from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, face_net, age_net, min_conf=0.5):

	# Define the list of age buckets our age detector will predict
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

	# Initialize results list
	results = []

	# Grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# Pass the blob through the network and obtain the face detections
	face_net.setInput(blob)
	detections = face_net.forward()

	# Loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > min_conf:
			# Compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			startX, startY, endX, endY = box.astype("int")
			# extract the ROI of the face
			face = frame[startY:endY, startX:endX]

			# ensure the face ROI is sufficiently large
			# To filter out false-positive face detections in frame and age classification won't be accurate for faces that are far away from the camera
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

			# Make prediction on the age and find the age bucket with the largest corresponding probability
			age_net.setInput(faceBlob)
			preds = age_net.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			age_confidence = preds[0][i]

			# Construct a dictionary consisting of both the face bounding box location along with the age prediction, then update our results list
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, age_confidence)
			}

			results.append(d)

	return results

# # Code for parsing arguments from terminal
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to input image")
# ap.add_argument("-f", "--face", required=True, help="Path to face detector model directory")
# ap.add_argument("-a", "--age", required=True, help="Path to age detector model directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")

# args = vars(ap.parse_args)

# Temporary variables for testing
args = {"face": "./face_detector", "age": "./age_detector", "image": "images/gadot.png", "confidence": 0.5}

# Load pretrained Face detector model
print("[INFO] Loading face detector model...")
face_prototxt_path = os.path.join(args["face"], "face_deploy.prototxt")
face_weights_path = os.path.join(args["face"], "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNet(face_prototxt_path, face_weights_path)

# Load pretrained age detection model
print("[INFO] Loading age detector model...")
age_prototxt_path = os.path.join(args["age"], "age_deploy.prototxt")
age_weights_path = os.path.join(args["age"], "age_net.caffemodel")
age_net = cv2.dnn.readNet(age_prototxt_path, age_weights_path)

# Initialize the video stream and allow the camera sonsor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
	# Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(0)