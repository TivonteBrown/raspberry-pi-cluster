#!/usr/bin/env python
"""
Parallel Heat Distribution

"""

from mpi4py import MPI

from sys import argv
from time import time

from numpy import zeros

from mpi_utils import gather_data, get_num_rows
from utils import do_single_round, initialize_matrix, write_matrix_as_image

EPSILON = 1.0e-6
VERBOSE = False



def synchronous_data(data, process_id, number_of_processes, error, comm=MPI.COMM_WORLD):
    row = data[0]
    # if not process_id == 0 then send row 1 to previous process and receive from previous process row 0
    
    if process_id !=0:
        
        #comm.send (row[1], dest = process_id - 1) #send row 1 to previous process
        #data = comm.recv(row[0], source = process_id - 1) #receive row 0 from previous process
    
        comm.Sendrecv(data[1], dest = process_id - 1, \
                  recvbuf = data[0], source = process_id - 1)
    
    # if not process_id == number_of_processes - 1 then send second to last row to next process and recieve from next process the last row
    # note data [-2] is the second to last row and data [-1] is the lasmpirun.openmpi -np 2 -machinefile /home/pi/mpi4py/machinefile python parallel_heat_distribution.pyt row

    if process_id != number_of_processes - 1:
        #comm.send (row[-2], dest = process_id + 1)
        #data = comm.recv(row[-1], source = process_id + 1 )
        
        comm.Sendrecv(data[-2], dest = process_id + 1, \
                     recvbuf = data[-1], source = process_id + 1)

    # also need to know error so
    # reduce all error to root (process with rank 0) with the MPI.MAX operation

    
    error = comm.reduce(error, MPI.MAX)
    
    # and broadcast from the root the reduced maxmimum error
    error = comm.bcast(error, root=0)

    
    # return the error so we can use it to determine if we should stop
    
    return error

def process(matrix_dimension, process_id, number_of_processes, verbose=False):
    num_rows = get_num_rows(matrix_dimension, process_id, number_of_processes) + 2
    data = zeros((num_rows, matrix_dimension))
    error = 1.0 # to get the loop going, any value greater then EPSILON will work
    initialize_matrix(data, process_id, number_of_processes)
    iteration = 0
    while error > EPSILON:
        # do a single iteration
        
        error = do_single_round(data)
        # synchronous the data
        error = synchronous_data(data, process_id, number_of_processes, error, comm=MPI.COMM_WORLD)
        
        
        iteration += 1
        if verbose and process_id == 0:
            print(f'Iteration {iteration}: error = {error:.6f}')
    return data

def main():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    number_of_processes = comm.Get_size()
    #print (argv)
    matrix_dimension = int(argv[1])
    output_filename = argv[2]
    start = time()
    partial = process(matrix_dimension, process_id, number_of_processes, verbose=VERBOSE)
    result = gather_data(matrix_dimension, process_id, number_of_processes, partial)
    if process_id == 0:
        elapsed = time() - start
        write_matrix_as_image(output_filename, result)
        print(f'({elapsed:.3f})', end='')

main()
