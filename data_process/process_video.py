import os
from tqdm import tqdm
import shutil


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
            if os.path.isdir(video_path):
                pass
            else:
                self.video_name = name.split('.')[0]
                print(os.getcwd())
                cmd = "python3 video_demo.py --video {} --outdir {} --save_video --sp".format(video_path, self.buffer_path)
                os.system(cmd)
                self.buffer2output()
        os.removedirs(self.buffer_path)


if __name__ == '__main__':
    input_path = ''
    VP = VideoProcessor(input_path)
    VP.run_video()

