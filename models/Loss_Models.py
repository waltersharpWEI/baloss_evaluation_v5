import random
import numpy as np
#this is to generate the loss matrix for the stack bar chart
def get_loss_matrix(size):
    matrix = np.random.randint(0,10, [size,size]).astype(np.float32)
    matrix = matrix / 10.0
    for i in range(size):
        val = 1 - matrix[i, 0] - matrix[i, 1]
        while val < 0:
            matrix[i,0] = random.random()
            matrix[i,1] = random.random()
            val = 1 - matrix[i, 0] - matrix[i, 1]
        matrix[i,2] = val
    print(matrix)
    return matrix

if __name__=="__main__":
    get_loss_matrix(3)