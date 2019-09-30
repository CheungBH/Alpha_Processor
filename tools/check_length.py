import os


def get_line(path):
    file = open(path, "r")
    lines = [line for line in file.readlines()]
    return len(lines)


def check_equal(folder):
    data_path = os.path.join(folder, "data.txt")
    label_path = os.path.join(folder, "label.txt")
    if get_line(data_path) == get_line(label_path):
        return True
    else:
        return False


if __name__ == '__main__':
    file_folder = '../network/test3'
    print(check_equal(file_folder))