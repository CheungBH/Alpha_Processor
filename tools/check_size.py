import numpy as np

txt_path = r'C:\Users\hkuit164\Desktop\golf0213_driving\36points\cmu_640x480\data_5\DFNoFinish\p1\00.txt'

file_matrix = np.loadtxt(txt_path)
print("The file has {} rows".format(file_matrix.shape[0]))
print("Each row has {} numbers".format(file_matrix.shape[1]))