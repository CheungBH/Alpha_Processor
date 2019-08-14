import os

file_path = r'C:\Users\hkuit164\Desktop\AlphaPose-pytorch\temp\sport00\zzz_data\all\label.txt'
file = open(file_path, "r")
lines = []
for line in file.readlines():
    lines.append(line)
    print(line)

print(len(lines))