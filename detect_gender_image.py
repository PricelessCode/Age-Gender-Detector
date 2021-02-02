import numpy as np
import cv2
import argparse
import os


# # Code for parsing arguments from terminal
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to input image")
# ap.add_argument("-f", "--face", required=True, help="Path to face detector model directory")
# ap.add_argument("-a", "--age", required=True, help="Path to age detector model directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")

# args = vars(ap.parse_args)

# Temporary variables for testing
args = {"face": "./face_detector", "gender": "./gender_detector", "image": "images/old_woman.png", "confidence": 0.5}

GENDER_BUCKET = ['M', 'F']

# Load pretrained Face detector model
print("[INFO] Loading face detector model...")
face_xml = os.path.join(args["face"], "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(face_xml)

# Load pretrained age detection model
print("[INFO] Loading gender detector model...")
gender_prototxt_path = os.path.join(args["gender"], "gender_deploy.prototxt")
gender_weights_path = os.path.join(args["gender"], "gender_net.caffemodel")
gender_net = cv2.dnn.readNet(gender_prototxt_path, gender_weights_path)

# Load the input image and construct a gray scaled image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detections = face_cascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE
)

# Loop over detections
for x, y, w, h in detections:

	# Extract the ROI of the face and construct a blob from only the face ROI
	face = image[y:y+h, x:x+w]
	
	faceBlob =cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
		swapRB=False)

	# # Make Predictions on gender and find the age bucket with the largest corresponding probability
	gender_net.setInput(faceBlob)
	preds = gender_net.forward()
	i = preds[0].argmax()
	gender = GENDER_BUCKET[i]
	gender_confidence = preds[0][i]

	# Display the predicted age to the terminal
	text = "{}: {:.2f}%".format(gender, gender_confidence * 100)
	print("[INFO] {}".format(text))

	# Draw the bounding box of the face along with the associated predicted age
	cv2.rectangle(image, (x, y), (x+h, y+h), (0, 255, 0), 2)
	cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()