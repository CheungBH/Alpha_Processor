import numpy as np

txt_path = '../network/test4/data.txt'

file_matrix = np.loadtxt(txt_path)
print("The file has {} rows".format(file_matrix.shape[0]))
try:
    print("Each row has {} numbers".format(file_matrix.shape[1]))
except IndexError:
    print("Each row has 1 number")