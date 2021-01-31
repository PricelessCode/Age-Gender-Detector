from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, face_cascade, age_net):

	# Define the list of age buckets our age detector will predict
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

	GENDER_BUCKET = ['\u2642', '\u2640']

	# Initialize results list
	results = []

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	detections = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	# Loop over the detections
	for x, y, w, h in detections:
		
		face = frame[y:y+h, x:x+w]

		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

		# Make prediction on the age and find the age bucket with the largest corresponding probability
		age_net.setInput(faceBlob)
		preds = age_net.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		age_confidence = preds[0][i]

		# Make prediction on the age and find the age bucket with the largest corresponding probability
		gender_net.setInput(faceBlob)
		preds = gender_net.forward()
		i = preds[0].argmax()
		gender = GENDER_BUCKET[i]
		gender_confidence = preds[0][i]

		# Construct a dictionary consisting of both the face bounding box location along with the age prediction, then update our results list
		d = {
			"loc": (x, y, x + h, y + h),
			"age": (age, age_confidence),
			"gender": (gender, gender_confidence)
		}

		results.append(d)

	return results

# # Code for parsing arguments from terminal
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to input image")
# ap.add_argument("-f", "--face", required=True, help="Path to face detector model directory")
# ap.add_argument("-a", "--age", required=True, help="Path to age detector model directory")

# args = vars(ap.parse_args)

# Temporary variables for testing
args = {"face": "./face_detector", "age": "./age_detector", "gender": "./gender_detector"}

# Load pretrained Face detector model
print("[INFO] Loading face detector model...")
face_xml = os.path.join(args["face"], "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(face_xml)

# Load pretrained age detection model
print("[INFO] Loading age detector model...")
age_prototxt_path = os.path.join(args["age"], "age_deploy.prototxt")
age_weights_path = os.path.join(args["age"], "age_net.caffemodel")
age_net = cv2.dnn.readNet(age_prototxt_path, age_weights_path)

# Load pretrained gender detection model
print("[INFO] Loading gender detector model...")
gender_prototxt_path = os.path.join(args["gender"], "gender_deploy.prototxt")
gender_weights_path = os.path.join(args["gender"], "gender_net.caffemodel")
gender_net = cv2.dnn.readNet(gender_prototxt_path, gender_weights_path)

# Initialize the video stream and allow the camera sonsor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
	# Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detect faces in the frame, and for each face in the frame, predict the age
	results = detect_and_predict_age(frame, face_cascade, age_net)

	key = None
	# Loop over the results
	for r in results:
		
		# Age precition Text for the bounding box
		age_text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)

		# Draw the bounding box of the face along with the associated predicted age
		startX, startY, endX, endY = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

		cv2.putText(frame, age_text, (startX, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.45, (0, 0, 255), 2)

		# Text for the bounding box
		gender_text = "{}: {:.2f}%".format(r["gender"][0], r["gender"][1] * 100)

		# Gender prediction text
		cv2.putText(frame, gender_text, (startX, y - 25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.45, (255, 0, 0), 2)
	
	# Show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
	
# Do clean up	
cv2.destroyAllWindows()
vs.stop()
