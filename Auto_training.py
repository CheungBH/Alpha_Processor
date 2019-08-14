from config import config
from train.trainer import RunNetwork

networks = config.network_structure_ls
epochs = config.epoch_ls
dropouts = config.dropout_ls
val_ratios = config.val_ratio_ls
class_name = config.class_name
X_Vector = config.X_vector
time_step = config.training_frame
data_info = config.data_info
iteration = config.begin_num

if __name__ == '__main__':
    for epoch in epochs:
        for dropout in dropouts:
            for net in networks:
                for val in val_ratios:
                    RunNetwork(epoch, dropout, net, class_name, X_Vector, val, time_step, iteration, data_info)
                    iteration = iteration + 1