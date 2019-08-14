import os

file_path = '../Data/Test/all/label.txt'
file = open(file_path, "r")
lines = []
for line in file.readlines():
    lines.append(line)
    print(line)

print(len(lines))