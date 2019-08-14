

def write_label(data_path, num):
    label_path = data_path.replace(".txt", "_label.txt")
    data_file = open(data_path, "r")
    lines = []
    for line in data_file.readlines():
        lines.append(line)
    data_file.close()
    length = len(lines)
    label_file = open(label_path, "w")
    for i in range(length):
        label_file.write(str(num))
        label_file.write('\n')
    label_file.close()


if __name__ == '__main__':
    data = '../Data/origin.txt'
    cnt = 126
    write_label(data, cnt)