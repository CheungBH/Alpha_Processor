from config import config
import os

from data_process.process_video import VideoProcessor
from data_process.process_json import JsonScanner
from data_process.process_txt import txtProcessor, txtMerger
from tools.final_merge import merge
from tools.label import write_label

main_folder = config.folder_path
class_ls = os.listdir(main_folder)
step = config.step
frame = config.frame
data_folder = os.path.join("data", config.action, config.folder, "{}frames_{}step".format(frame, step))


if __name__ == '__main__':
    print("Begin to process the videos in {}".format(main_folder))
    cnt = 1
    for cls in class_ls:
        print("{}/{}".format(cnt, len(class_ls)))
        print('Begin to process the videos of {}'.format(cls))
        action_folder = os.path.join(main_folder, cls)
        os.makedirs(data_folder, exist_ok=True)

        # Video to json
        act_VP = VideoProcessor(action_folder)
        act_VP.run_video()
        print("Finish processing the videos of {}".format(cls))

        print("Begin to transfer json to normalized coordinate")
        json_path = act_VP.output_path
        coord_path = os.path.join(json_path, "coordinate")
        os.makedirs(coord_path, exist_ok=True)
        # json to (txt)coordinate
        act_JS = JsonScanner(json_path, coord_path)
        act_JS.run()
        print("Finish transferring to coordinate")

        print("Begin to transfer coordinate into the form of input data")
        # (txt)coordinate to (txt)data
        txt_folder = os.path.join(data_folder, cls)
        act_txtP = txtProcessor(coord_path, txt_folder, step, frame)
        act_txtP.run()
        data_path = os.path.join(data_folder, "{}.txt".format(cls))
        print('Finish transferring from coordinate to input data')

        print("Begin to merge the input data of {}".format(cls))
        # Merge data within a class
        txtM = txtMerger(txt_folder, data_path)
        txtM.run()

        print("Finish merging; Begin to produce label")

        # Write the label
        write_label(data_path, cnt)
        cnt += 1

        print("Finish processing {}".format(cls))
        print("\n")
        print("\n")

    merge(data_folder)