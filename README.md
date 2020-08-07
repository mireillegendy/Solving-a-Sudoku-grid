The purpose of this project is to experiment with CNNs. The motive of this project was to use the digit
recognition component of the project and utilise the findings to solve a sudoku puzzle.
In a nutshell this program processes an input image, of an unsolved sudoku grid with hand written digits.
The algorithm then solves the puzzle and displays the solution in another grid.
The project is divided in three main parts:
1) The algorithm to solve the grid:
This portion comprises of the core algorithm that solves the sudoku puzzle. The algorithm accepts any 
unsolved grid with hand written digits, and displays the answer. The code to this part of the project 
is in sudoku_algorithm.py file.
2) The image processing component of the algorithm
The second portion of the project analyzes the input image of the sudoku grid to extract the digits from 
the unsolved grid and replaces the empty squares with zero.
3) The convolutional neural network to understand the hand written digits
The last component of the algorithm is a CNN that recognizes the hand written digits and converts them to
 digits the computer can comprehend. 
 
To run the project: 
import an image of the unsolved sudoku grid, with hand written digits, to the root project directory and 
change the image name accordingly in the main file on line 1.

After running the code successfully the solved puzzle will be displayed in a new window.

References: 
1) https://norvig.com/sudoku.html ---> Algorithm to solve the Sudoku puzzle
2) https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a ---> CNN Model
 Architecture for digit recognition
3) https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2 ---> Image processing portion to 
extract digits and manipulate input image



