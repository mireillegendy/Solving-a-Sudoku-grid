import cv2
from image_processing import pre_processing, find_corners, fix_tilt, divide_grid, get_digits, img_digit, data_set_pre_processing, format_puzzle, display_digits
import tensorflow as tf
from cnn_model import x_train_raw, y_train_raw, CNN_model
import numpy as np
from sudoku_algorithm import solve

image_title = "hand_written.png"
img = cv2.imread(image_title, cv2.IMREAD_GRAYSCALE)
processed = pre_processing(img)
corners = find_corners(processed)
cropped = fix_tilt(img, corners)
squares = divide_grid(cropped)
digits = get_digits(cropped, squares, 28)
zero_img = None
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
read_digits = []

for i in range(0, len(digits)):
    if cv2.countNonZero(digits[i]) == 0:
        read_digits.append(img_digit(0))
    else:
        read_digits.append(digits[i])
array_digits = np.array(read_digits)
array_digits_raw = array_digits
array_digits = data_set_pre_processing(array_digits)
predictions = CNN_model.predict([array_digits])
puzzle = ''
for i in range(0, len(read_digits)):
  puzzle = puzzle + str(np.argmax(predictions[i]))

puzzle = format_puzzle(puzzle)
print(puzzle)
solved = solve(puzzle)
puzzle_list = list(solved.values())
for i in range(0, len(puzzle_list)):
    puzzle_list[i] = int(puzzle_list[i])
solved_digit_image_list = []
for i in range(0, len(puzzle_list)):
    solved_digit_image_list.append(img_digit(puzzle_list[i]))
display_digits(solved_digit_image_list, 255)
