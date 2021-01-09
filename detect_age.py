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