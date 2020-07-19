import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
def show_image(img):

	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	img = in_img.copy()
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

def scale(x,r): return x*r


def scale_centre(img, size, margin=0, background=0):
    h, w = img.shape
    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size-length)/2)
            side2=side1
        else:
            side1 = int((size - length)/2)
            side2 = side1+1
        return side1, side2
    if h > w:
        t_pad = int(margin/2)
        b_pad = t_pad
        ratio = (size-margin)/h
        h,w = int(scale(ratio, h)), int(scale(ratio, w))
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin/2)
        r_pad = l_pad
        ratio = (size-margin) / w
        h, w = int(scale(ratio, h)), int(scale(ratio, w))
        t_pad, b_pad = centre_pad(h)
    img = cv2.resize(img, (w,h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))






img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
result = pre_processing(img)
corners = find_corners(result)
alligned_result = fix_tilt(result, corners)
img1 = scale_centre(alligned_result, 500)
show_image(img1)
h,w = img.shape
print(w)