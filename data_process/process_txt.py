import os

input_path = r'C:\Users\hkuit164\Desktop\AlphaPose-pytorch\temp\test'
output_path = r'C:\Users\hkuit164\Desktop\AlphaPose-pytorch\temp\test_result'
time_step = 3
frame = 5


class txtProcessor(object):
    def __init__(self, coord_path, txt_path, timestep, frame):
        self.coord_path = coord_path
        self.txt_path = txt_path
        os.makedirs(self.txt_path, exist_ok=True)
        self.content = []
        self.timestep = timestep
        self.frame = frame
        self.file_name = ''
        self.result = []

    def process_txt(self):
        for begin_line in range(len(self.content))[:-self.frame + 1:self.timestep]:
            result = ''
            for num in range(self.frame):
                frame_res = self.content[begin_line + num]
                result = result + frame_res.replace('\n', ' ')
            # if result[-1] == ' ':
            #     result = result[:-1]
            self.result.append(result[:-1] + '\n')

    def write_result(self):
        dest_path = os.path.join(self.txt_path, self.file_name)
        dest_file = open(dest_path, "w")
        for line in self.result:
            dest_file.write(str(line))
        dest_file.write('\n')
        dest_file.close()

    def run(self):
        for self.file_name in os.listdir(self.coord_path):
            self.content = []
            self.result = []
            file = open(os.path.join(self.coord_path, self.file_name), 'r')
            for line in file.readlines():
                self.content.append(line)
            self.process_txt()
            self.write_result()


class txtMerger(object):
    def __init__(self, src_folder, dest_path):
        self.src_folder = src_folder
        self.dest_path = dest_path
        # self.sample_cnt = 0
        self.content = []

    def read_file(self):
        for file_name in os.listdir(self.src_folder):
            file = open(os.path.join(self.src_folder, file_name), "r")
            for line in file.readlines():
                self.content.append(line)
            try:
                self.content.remove('\n')
            except ValueError:
                pass
            file.close()

    def write_file(self):
        data_file = open(self.dest_path, 'w')
        for line in self.content:
            data_file.write(line)

    def run(self):
        self.read_file()
        self.write_file()


class MergeAndLabel(object):
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.sample_cnt = 0
        self.name = ''
        self.dest_path = ''
        self.action = os.listdir(self.main_folder)
        try:
            self.action.remove("zzz_data")
        except ValueError:
            pass
        self.label_num = len(self.action)
        # os.makedirs(self.dest_folder, exist_ok=True)
        # self.data_path = os.path.join(self.dest_folder, "data.txt")
        # self.label_path = os.path.join(self.dest_folder, "label.txt")
        self.content = []
        self.txt_path = ''

    def read_file(self):
        for file_name in os.listdir(self.txt_path):
            file = open(os.path.join(self.txt_path, file_name), "r")
            for line in file.readlines():
                self.content.append(line)
            try:
                self.content.remove('\n')
            except ValueError:
                pass
            file.close()

    def write_file(self):
        data_file = open(self.dest_path, 'w')
        for line in self.content:
            data_file.write(line)
        self.sample_cnt = len(self.content)

    def label(self, num):
        label_path = self.dest_path.replace(".txt", "_label.txt")
        label_file = open(label_path, "w")
        for i in range(self.sample_cnt):
            label_file.write(str(num))
            label_file.write('\n')

    def run(self):
        dest_folder = os.path.join(self.main_folder, "zzz_data")
        os.makedirs(dest_folder, exist_ok=True)
        for action, label_num in zip(self.action, range(self.label_num)):
            self.txt_path = os.path.join(self.main_folder, action)
            self.dest_path = os.path.join(dest_folder, action + ".txt")
            self.read_file()
            self.write_file()
            self.label(label_num + 1)
            self.content = []


if __name__ == '__main__':
    # txtP = txtProcessor(input_path, output_path, time_step, frame)
    # # txtP.run()
    # path = r'C:\Users\hkuit164\Desktop\AlphaPose-pytorch\temp\sport00'
    # ML = MergeAndLabel(path)
    # ML.run()
    src = '../Data/origin'
    dest = '../Data/origin.txt'
    txtM = txtMerger(src, dest)
    txtM.run()