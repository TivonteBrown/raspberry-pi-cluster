from sys import argv
from time import time

from numpy import zeros
from utils import do_single_round, initialize_matrix, write_matrix_as_image

EPSILON = 1.0e-6
VERBOSE = False

def process(matrix_dimension, verbose=False):
    data = zeros((matrix_dimension, matrix_dimension))
    error = 1.0 # to get the loop going, any value greater then EPSILON will work
    initialize_matrix(data, 0, 1)
    iteration = 0
    while error > EPSILON:
        error = do_single_round(data)
        iteration += 1
        if verbose:
            print(f'Iteration {iteration}: error = {error:.6f}')
    return data

def main():
    matrix_dimension = int(argv[1])
    output_filename = argv[2]
    start = time()
    result = process(matrix_dimension, verbose=VERBOSE)
    elapsed = time() - start
    write_matrix_as_image(output_filename, result)
    print(f'({elapsed:.3f})', end='')

main()
