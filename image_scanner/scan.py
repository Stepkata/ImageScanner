from transform import four_point_transform
from skimage.filters import threshold_local
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
#clone and resize it to make edge detection more accurate
image = cv.imread(args["image"])
ratio = image.shape[0]/ 500.0
orig = image.copy()
image = imutils.resize(image, height=500) 

#convert the image to grayscale, blur it, and find edges 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(gray, 75, 200)

#show the original image and the edge detected image
print("Step 1: Edge detection")
cv.imshow("Image", image)
cv.imshow("Edged", edged)
cv.waitKey(0)
cv.destroyAllWindows()

#finde the contours in the edged image, keeping only the largest
#ones, and initialise screen contour
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

#loop over the contours
for c in cnts:
    #approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    #if our approximated contour has four points, then we
    #can assume we have found the screen
    if len(approx) == 4:
        screenCnt = approx
        break


#show the outline of the piece of paper
print("STEP 2: Find contours of paper")
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)
cv.destroyAllWindows()

#apply the four point transform to obtain a top-down view
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv.imshow("Original", imutils.resize(orig, height = 650))
cv.imshow("Scanned", imutils.resize(warped, height = 650))
cv.waitKey(0)
