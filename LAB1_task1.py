
"""TASK 1 : matrix multiplication executed iteratively by increasing the size n by one each iteration"""""

#importing libraries (numpy for matrix, time for the measure of the execution time, matplotlib.pyplot to draw the diagram of the execution time) 
import numpy as np
import time
import matplotlib.pyplot as plt

#Function matrix_multiply : the goal is to multiply 2 matrix executed iteratively by increasing the size n by one each iteration and to measure the time execution
def matrix_multiply(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    start_time = time.time()
    result = np.dot(A, B)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return execution_time

n_values = [30, 50, 80, 100, 150]
execution_times = []

for n in n_values:
    execution_time = matrix_multiply(n)
    execution_times.append(execution_time)
    print(f"Matrix size {n}x{n} took {execution_time:.4f} seconds to multiply.")

#Plot the diagram of the execution time function of n_values
plt.plot(n_values, execution_times, marker='D', linestyle='--')
plt.title("Matrix multiplication execution time depending to matrix Size")
plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Execution Time in seconds)")
plt.grid(True)
plt.show()

