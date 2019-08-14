import os
import cv2
import numpy as np
import pandas as pd


class JsonScanner(object):
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize
        input_dir: directory of json files with corresponding videos
        output_dir: directory of ouput txt fiels
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.json_paths = self.get_json_paths()
        self.video_paths, self.video_names = self.get_video_paths()
        self.index_dict = self.get_index_dict()

    def get_json_paths(self):
        json_paths = []
        for json_file in os.listdir(self.input_dir):
            if json_file.split('.')[-1] == 'json':
                path = os.path.join(self.input_dir, json_file)
                json_paths.append(path)
        return json_paths

    def get_video_paths(self):
        video_paths = []
        video_names = {}
        for video_file in os.listdir(self.input_dir):
            if video_file.split('.')[-1] != 'json':
                path = os.path.join(self.input_dir, video_file)
                video_paths.append(path)
                video_names[video_file.split('.')[0]] = path
        return video_paths, video_names

    def get_index_dict(self):
        index_dict = {}
        for json_path in self.json_paths:
            video_name = self.extract_file_id(json_path)
            index_dict[json_path] = self.video_names[video_name]

        return index_dict

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.release()
        return width, height

    def drop_scores(self, data_list):
        new_list = []
        for index, value in enumerate(data_list):
            if (index + 1) % 3 == 0:
                continue
            else:
                new_list.append(value)
        return new_list

    def normalize_coordinates(self, coordinates, w, h):
        new_coordinates = []
        for i in range(len(coordinates)):
            if (i + 1) % 2 == 0:
                new_coordinates.append(coordinates[i] / h)
            else:
                new_coordinates.append(coordinates[i] / w)
        return new_coordinates

    def extract_file_id(self, path):
        file_id = os.path.split(path)[-1]
        file_id = file_id.split('.')[0]

        return file_id

    def json2txt(self, json_path):
        df = pd.read_json(json_path, orient='dict')
        if len(df) != 0:
            df = self.drop_duplicated(df)
            keypoints = df['keypoints'].copy()
            w, h = self.get_video_info(self.index_dict[json_path])
            for i in range(len(keypoints)):
                keypoints[i] = self.drop_scores(keypoints[i])
                keypoints[i] = self.normalize_coordinates(keypoints[i], w, h)

            keypoints = np.array(keypoints.tolist())
            txt_name = self.extract_file_id(json_path) + '.txt'
            txt_path = os.path.join(self.output_dir, txt_name)
            np.savetxt(txt_path, keypoints, fmt='%.8f')
        else:
            pass

    def drop_duplicated(self, df):
        haha = df.copy()
        df = df.sort_values(by='score', ascending=False)
        df = df.drop_duplicates(subset='image_id', keep='first')
        df = df.sort_index()
        df = df.reset_index(drop=True)
        return df

    def run(self):
        for i in self.json_paths:
            self.json2txt(i)


if __name__ == '__main__':
    pass