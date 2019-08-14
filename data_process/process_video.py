import os
from tqdm import tqdm
import shutil
#
# input_dir = "Video/test"
# output_dir = os.path.join(input_dir, "output")
# buffer_dir = os.path.join(input_dir, "buffer")
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(buffer_dir, exist_ok=True)
input_path = ""


class VideoProcessor(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.output_path = os.path.join(self.video_path, "output")
        self.buffer_path = os.path.join(self.output_path, "buffer")
        self.video_name = ""
        os.makedirs(self.buffer_path, exist_ok=True)

    def buffer2output(self):
        for file_name in os.listdir(self.buffer_path):
            origin_path = os.path.join(self.buffer_path, file_name)
            if file_name.split('.')[-1] == 'json':
                dest_path = os.path.join(self.output_path, "AlphaPose_" + self.video_name + '.json')
            else:
                dest_path = os.path.join(self.output_path, "AlphaPose_" + self.video_name + '.avi')
            shutil.move(origin_path, dest_path)

    def run_video(self):
        for name in tqdm(os.listdir(self.video_path)):
            video_path = os.path.join(self.video_path, name)
            self.video_name = name.split('.')[0]
            cmd = "python video_demo.py --video {} --outdir {} --save_video".format(video_path, self.buffer_path)
            os.system(cmd)
            self.buffer2output()
        os.remove(self.buffer_path)


if __name__ == '__main__':
    VP = VideoProcessor(input_path)
    VP.run_video()

