import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
def show_image(img):
	"""Shows an image until any key is pressed"""
	cv2.imshow('image', img)  # Display the image
	cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
	cv2.destroyAllWindows()  # Close all windows
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	"""Draws circular points on an image."""
	img = in_img.copy()

	# Dynamically change to a colour image if necessary
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	for point in points:
		img = cv2.circle(img, tuple(x for x in point), radius, colour, -1)
	show_image(img)
	return img
def pre_processing(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9,9), 0)
    proc = cv2.adaptiveThreshold(proc,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc)
    if not skip_dilate:
        kernel = np.array(([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]), np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc
def find_corners(img):
    contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sort_corners = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = sort_corners[0]
    top_right, _ = max(enumerate(pt[0][0] - pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    top_left, _ = min(enumerate(pt[0][0] + pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate(pt[0][0] - pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    bottom_right, _ = max(enumerate(pt[0][0] + pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
def distance_between_points(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt(pow(a, 2) + pow(b, 2))

def fix_tilt(img, corners):
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max(distance_between_points(top_left, bottom_left),
               distance_between_points(top_left, top_right),
               distance_between_points(top_right, bottom_right),
               distance_between_points(top_right, bottom_right))
    dst = np.array([[0,0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
result = pre_processing(img)
corners = find_corners(result)
tilt = fix_tilt(result, corners)
show_image(tilt)