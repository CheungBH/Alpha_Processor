import os

file_path = '../data/Swimming/Test/5frames_2step/action1_label.txt'
file = open(file_path, "r")
lines = []
for line in file.readlines():
    lines.append(line)
    # print(line)
print(file_path)
print(len(lines))