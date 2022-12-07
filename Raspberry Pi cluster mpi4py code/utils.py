from math import pi, sin
from numpy import absolute, amax

def initialize_matrix(data, process_id, number_of_processes):
    if not process_id in (0, number_of_processes - 1):
        # nothing to initialize
        return

    if process_id == 0:
        row = data[0]
        cols = len(row) - 1
        for col in range(cols + 1):
            row[col] = 255.0 * sin(pi * col / cols)

    if process_id == number_of_processes - 1:
        row = data[-1]
        cols = len(row) - 1
        for col in range(cols + 1):
            row[col] = 255.0 * sin(pi * col / cols)

def do_single_round(data):
    temp = data.copy()
    rows, cols = data.shape
    data[1:rows-1, 1:cols-1] = (temp[0:rows-2, 1:cols-1] + temp[2:rows, 1:cols-1]\
                              + temp[1:rows-1, 0:cols-2] + temp[1:rows-1, 2:cols]) / 4.0
    return amax(absolute(data - temp))

def write_matrix_as_image(filename, matrix):
    image = open(filename, 'w')
    raw_file = open(filename + '.raw', 'w')
    shape = matrix.shape
    rows = shape[0]
    cols = shape[1]
    image.write(f'P6\n{rows} {cols}\n255\n')
    image.close()
    image = open(filename, 'ab')
    zero = 0
    ascii_zero = zero.to_bytes(1, 'big')
    for row in matrix:
        for cell in row:
            red = int(cell + 0.5)
            blue = 255 - red
            image.write(red.to_bytes(1, 'big'))
            image.write(ascii_zero)
            image.write(blue.to_bytes(1, 'big'))
            raw_file.write(f'{cell:.3f}\n')
    image.close()
    raw_file.close()
