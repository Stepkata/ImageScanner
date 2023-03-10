from transform import four_point_transform
from skimage.filters import treshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
                help = "Path to image to be scanned")
args = vars(ap.parse_args())

#load the image and compute the ratio of the old height to new height
#clone and resize it
image = cv.imread(args["image"])
ratio = image.shape[0]/ 500.0
orig = image.copy()
image = imutils.resize(image, height=500)