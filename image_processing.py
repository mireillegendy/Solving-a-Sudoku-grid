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
def find_largest_area(inp_img, scan_tl=None, scan_bl=None):
    img = inp_img.copy()
    max_area = 0
    seed_point = (None, None)
    height, width = img.shape
    if scan_tl == None:
        scan_tl = [0, 0]
    if scan_bl == None:
        scan_bl = [width, height]
    for x in range(scan_tl[0], scan_bl[0]):
        for y in range(scan_tl[1], scan_bl[1]):
            if img[y, x] == 255:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    for x in range(width):
        for y in range(height):
            if img[y, x] == 255:
                cv2.floodFill(img, None, (x, y), 64)
    mask = np.zeros((height + 2, width + 2), np.uint8)
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)
    top, bottom, right, left = height, 0, 0, width
    for x in range(width):
        for y in range(height):
            if img[y, x] == 64:
                cv2.floodFill(img, mask, (x, y), 0)
            if img[y, x] == 255:
                if x < left: left = x
                if x > right: right = x
                if y < top: top = y
                if y > bottom: bottom = y
    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point
def cut_rect_from_img(img, rect): #knowing that rect = [[topx, topy], [bottomx, bottomy]]
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def extract_digits(img, rect, size):
    digit = cut_rect_from_img(img, rect)
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w])/2.5)
    _, bbox, seed = find_largest_area(digit, [margin, margin], [w-margin, h-margin])
    digit = cut_rect_from_img(digit, bbox)
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    if w > 0 and h > 0 and (w*h) > 100 and len(digit) > 0:
        return scale_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)
def divide_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0]/9
    for i in range(9):
        for j in range(9):
            p1 = (i*side, j*side)
            p2 = ((i+1)*side, (j+1)*side)
            squares.append((p1, p2))
    return squares
def get_digits(img, squares, size):
    digits = []
    img = pre_processing(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digits(img, square, size))
    return digits
def display_digits(digits, colour = 255):
    rows = []
    with_borders = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT,None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_borders[i*9: (i+1)*9], axis=1)
        rows.append(row)
    return show_image(np.concatenate(rows))


img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
processed = pre_processing(img)
corners = find_corners(processed)
cropped = fix_tilt(img, corners)
squares = divide_grid(cropped)
digits = get_digits(cropped, squares, 28)
display_digits(digits)