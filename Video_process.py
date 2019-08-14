from config import config
import os
from data_process.process_video import VideoProcessor
from data_process.process_json import JsonScanner
from data_process.process_txt import txtProcessor, MergeAndLabel
from tools.final_merge import merge
import shutil

class_ls = config.class_ls
main_folder = config.folder_path
step = config.step
frame = config.frame
data_path = os.path.join("data", config.action, config.folder, "{}frames_{}timestep".format(frame, step))


if __name__ == '__main__':
    for cls in class_ls:
        action_folder = os.path.join(main_folder, cls)
        txt_path = os.path.join(action_folder, "result", cls)
        os.makedirs(txt_path, exist_ok=True)
        act_VP = VideoProcessor(action_folder)
        act_VP.run_video()
        coord_path = os.path.join(act_VP.output_path, "coord_file")
        act_JS = JsonScanner(act_VP.output_path, coord_path)
        act_JS.run()
        act_txtP = txtProcessor(coord_path, txt_path, step, frame)
        act_txtP.run()
    ML = MergeAndLabel(os.path.join(main_folder, "result"))
    ML.run()
    merge(os.path.join(main_folder, "result", "zzz_data"))
    data_storage_path = os.path.join(main_folder, "result", "zzz_data", "all")
    for file in data_storage_path:
        shutil.copy(file, data_path)



