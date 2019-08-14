from config import config
import os
from data_process.process_video import VideoProcessor
from data_process.process_json import JsonScanner
from data_process.process_txt import txtProcessor, txtMerger
from tools.final_merge import merge
from tools.label import write_label

class_ls = config.class_ls
main_folder = config.folder_path
step = config.step
frame = config.frame
data_folder = os.path.join("data", config.action, config.folder, "{}frames_{}step".format(frame, step))


if __name__ == '__main__':
    cnt = 1
    for cls in class_ls:
        action_folder = os.path.join(main_folder, cls)
        os.makedirs(data_folder, exist_ok=True)

        # Video to json
        act_VP = VideoProcessor(action_folder)
        act_VP.run_video()
        json_path = act_VP.output_path
        coord_path = os.path.join(json_path, "coord_file")
        os.makedirs(coord_path, exist_ok=True)

        # json to (txt)coordinate
        act_JS = JsonScanner(json_path, coord_path)
        act_JS.run()

        # (txt)coordinate to (txt)data
        txt_folder = os.path.join(data_folder, cls)
        act_txtP = txtProcessor(coord_path, txt_folder, step, frame)
        act_txtP.run()
        data_path = os.path.join(data_folder, "{}.txt".format(cls))

        # Merge data within a class
        txtM = txtMerger(txt_folder, data_path)
        txtM.run()

        # Write the label of the clas
        write_label(data_path, cnt)
        cnt += 1
    merge(data_folder)



