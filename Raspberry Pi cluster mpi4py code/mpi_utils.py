from mpi4py import MPI
import numpy

def get_num_rows(matrix_dimension, process_id, number_of_processes):
    number_of_rows_needed_calculation = matrix_dimension - 2
    quotion = number_of_rows_needed_calculation // number_of_processes
    remainder = number_of_rows_needed_calculation % number_of_processes
    return quotion + (1 if process_id < remainder else 0)

def _get_sizes_array(number_of_processes, matrix_dimension):
    sizes = numpy.array([get_num_rows(matrix_dimension, pid, number_of_processes) \
                         for pid in range(number_of_processes)])
    sizes[0] += 1
    sizes[-1] += 1
    return sizes * matrix_dimension

def _get_offsets_array(number_of_processes, matrix_dimension, sizes):
    offsets = numpy.zeros(number_of_processes, dtype=numpy.int64)
    offsets[0] = 0
    for process_id in range(1, number_of_processes):
        offsets[process_id] = int(offsets[process_id - 1] + sizes[process_id - 1])
    return offsets

def gather_data(matrix_dimension, process_id, number_of_processes, local_matrix):
    sizes = _get_sizes_array(number_of_processes, matrix_dimension)
    offsets = _get_offsets_array(number_of_processes, matrix_dimension, sizes)
    result = numpy.zeros((matrix_dimension, matrix_dimension)) if process_id == 0 else None
    start = 0 if process_id == 0 else 1
    end = len(local_matrix) - (0 if process_id == number_of_processes - 1 else 1)
    localbuf = local_matrix[start:end]
    MPI.COMM_WORLD.Gatherv(localbuf, (result, sizes, offsets, MPI.DOUBLE), root=0)
    return result
