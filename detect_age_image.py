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
args = {"face": "./face_detector", "age": "./age_detector", "image": "images/gadot.png", "confidence": 0.5}

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]

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

# Load the input image and construct an input blob for the image
image = cv2.imread(args["image"])
(h, w) = image.shape[:2] # image.shape looks like (height, width, n of channels)
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the face detections
print("[INFO] computing face detections")
face_net.setInput(blob)
detections = face_net.forward()

# Loop over detections
for i in range(detections.shape[2]):
	# Extract confidence
	confidence = detections[0, 0, i, 2]

	# Make sure the detection confidence is higher than the minimum confidence
	if confidence > args["confidence"]:
		# Extract X, Y coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		startX, startY, endX, endY = box.astype("int")

		# Extract the ROI of the face and construct a blob from only the face ROI
		face = image[startY:endY, startX:endX]
		
		# # Code for checking the extracted face images.
		# cv2.imshow("Face Image", face)
		# cv2.waitKey(0)
		
		faceBlob =cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
			swapRB=False)

		# # Make Predictions on the age and find the age bucket with the largest corresponding probability
		age_net.setInput(faceBlob)
		preds = age_net.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		age_confidence = preds[0][i]

		# Display the predicted age to the terminal
		text = "{}: {:.2f}%".format(age, age_confidence * 100)
		print("[INFO] {}".format(text))

		# Draw the bounding box of the face along with the associated predicted age
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
		