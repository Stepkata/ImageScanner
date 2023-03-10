import numpy as np
import cv2 as cv

def order_points(pts):
    """initialise a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second top-right, the third bottom-right and the fourth
    entry is the bottom-left"""
    rect = np.zeros((4, 2), dtype = "float32")

    """top-left point will have the smallest sum
    bottom-right will have the largest sum"""
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    """compute the difference between the points, the
    top-right point will have the sallest difference,
    whereas the bottom-left will have the larges difference"""
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    """return the ordered coordinates"""
    return rect


def four_point_transform(image, pts):
    (tl, tr, br, bl) = order_points(pts) #obtain consistent order of the points

    """compute the width of new image, which will be the 
    maximum distance between bottom-right and bottom-left
    x-coordinates and top-right and top-left x-coordinates"""
    widthB = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthT = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    newWidth = max(int(widthB), int(widthT))

    """compute the height of new image, which will be the 
        maximum distance between top-right and bottom-right
        y-coordinates and top-left and bottom-left y-coordinates"""
    heightR = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightL = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    newHeight = max(int(heightL), int(heightR))

    """construct the set of destination points to obtain top-down view 
    of the image, specifying points in order just like in @order_points()"""
    dst = np.array(
        [0, 0],
        [newWidth -1, 0],
        [newWidth -1, newHeight-1],
        [0, newHeight-1],
        dtype="float32"
    )

    """compute the perspective matrix and then apply it"""
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (newWidth, newHeight))

    #return the warped image
    return warped




