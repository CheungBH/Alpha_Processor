import os

# For video process
action = "swimming"
folder = ""
folder_path = os.path.join('Video', action, folder)
class_ls = os.listdir(folder_path)
# For txt process
step = 2
frame = 5

# For auto_training
batch_size = 32
activation = 'relu'
optimizer = 'Adam'
data_path = ''  # Where the "label.txt" and the "data.txt" stored

epoch_ls = [1000]
dropout_ls = [0.1]
network_structure_ls = [1]
val_ratio_ls = [0.2]
class_name = ["Backswing", "Standing", "Final", "Downswing"]
X_vector = 36
training_frame = 5
data_info = "..."
begin_num = 1

if __name__ == '__main__':
    pass