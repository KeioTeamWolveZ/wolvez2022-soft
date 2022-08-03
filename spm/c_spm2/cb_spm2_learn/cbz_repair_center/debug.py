import numpy as np

def moving_average(x, num=10):
    ave_data = np.convolve(x, np.ones(num)/num)
    return ave_data

input_array=[1,2]
print(moving_average(input_array,num=2)[:-1])