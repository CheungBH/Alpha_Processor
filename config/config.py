import os

# For video process
action = "Swimming"
folder = "test"
folder_path = os.path.join('Video', action, folder)
# class_ls = os.listdir(folder_path)
# For txt process
step = 2
frame = 5

# For auto_training
batch_size = 32
activation = 'relu'
optimizer = 'Adam'
data_path = 'network/test3'  # Where the "label.txt" and the "data.txt" stored

epoch_ls = [100]
dropout_ls = [0.1]
network_structure_ls = [0]
val_ratio_ls = [0.2]

class_name = ["Backswing", "Standing", "Final", "Downswing"]
X_vector = 36
training_frame = 15
data_info = "test for openpose"
begin_num = 1

if __name__ == '__main__':
    pass