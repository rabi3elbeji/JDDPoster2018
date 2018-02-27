############################################################################
####################  The training script  #################################
############################################################################

# import of libraries
import os
from time import time
import numpy as np
import json
from models.model import Model
from utils.data_manager import DataManager
from utils.plot_data import PlotData
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json
from keras.callbacks import TensorBoard
# Use tensorflow as backend for keras
from keras import backend as K
K.set_image_dim_ordering('tf')


# Model type
model_type = 'wow_128_05'
last_model_type = 'wow_128_06'

# test dataset img
model_dataset = 'dataset_' + model_type

# Dataset dir paths
train_data_dir = './datasets/' + model_dataset + '/train'
validation_data_dir = './datasets/' + model_dataset + '/validation'


# Images width, height, channels
img_height = 128
img_width = 128
num_channels = 1

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
image_shape = (img_height, img_width, num_channels)

# Class Number
class_number = 2

# model ==> output paths
model_png = './trained_for_pred/' + model_type + '/model/model.png'
model_summary_file = './trained_for_pred/' + \
    model_type + '/model/model_summary.txt'
saved_model_arch_path = './trained_for_pred/' + \
    model_type + '/model/model.json'
saved_model_classid_path = './trained_for_pred/' + \
    model_type + '/model/model_classid.json'
train_log_path = './trained_for_pred/' + \
    model_type + '/model/log/model_train.csv'
train_checkpoint_path = './trained_for_pred/' + model_type + \
    '/model/log/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5'
model_tensorboard_log = './training_log/tensorbord/'


# model training params
num_of_epoch = 200
num_of_train_samples = 3400
num_of_validation_samples = 600


# Cost function
model_loss_function = 'binary_crossentropy'


# define optimizers
model_optimizer_rmsprop = 'rmsprop'
model_optimizer_adam0 = 'adam'
model_optimizer_adam = Adam(lr=0.003, decay=0.00001)
model_optimizer_sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)


best_weights = 'trained_for_pred/' + last_model_type + '/model/Best-weights.h5'

# model metrics to evaluate training
model_metrics = ["accuracy"]

# batch size
train_batch_size = 16
val_batch_size = 32

# for deleting a file


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass

# for saving model summary into a file


def save_summary(s):
    with open(model_summary_file, 'a') as f:
        f.write('\n' + s)
        f.close()
        pass

# predefined weights for preprocessing
KV = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12

KM = np.array([[0, 0, 5.2, 0, 0], [0, 23.4, 36.4, 23.4, 0], [
    5.2, 36.4, -261, 36.4, 5.2], [0, 23.4, 36.4, 23.4, 0], [0, 0, 5.2, 0, 0]], dtype=np.float32) / 261

GH = np.array([[0.0562, -0.1354, 0, 0.1354, -0.0562], [0.0818, -0.1970, 0, 0.1970, -0.0818], [0.0926, -0.2233, 0,
                                                                                              0.2233, -0.0926], [0.0818, -0.1970, 0, 0.1970, -0.0818], [0.0562, -0.1354, 0, 0.1354, -0.0562]], dtype=np.float32)
GV = np.fliplr(GH).T.copy()


'''
F0 = np.array([[17821374.0, 79869856.0, 131683128.0, 79869856.0, 17821374.0], [74341520.0, 333175552.0, 549313664.0, 333175552.0, 74341520.0], [0.0, -0.0, -0.0, -0.0, -0.0], [-74341520.0, -333175552.0, -549313664.0, -333175552.0, -74341520.0], [-17821374.0, -79869856.0, -131683128.0, -79869856.0, -17821374.0]], dtype=np.float32) 

F1 = np.array([[18870814.0, 84573120.0, 139437504.0, 84573120.0, 18870814.0], [70552608.0, 316194848.0, 521317184.0, 316194848.0, 70552608.0], [0.0, -0.0, -0.0, -0.0, -0.0], [-70552608.0, -316194848.0, -521317184.0, -316194848.0, -70552608.0], [-18870814.0, -84573120.0, -139437504.0, -84573120.0, -18870814.0]], dtype=np.float32) 

F2 = np.array([[-19544258.0, -87591288.0, -144413616.0, -87591288.0, -19544258.0], [-65697424.0, -294435424.0, -485441952.0, -294435424.0, -65697424.0], [0.0, 0.0, 0.0, 0.0, 0.0], [65697424.0, 294435424.0, 485441952.0, 294435424.0, 65697424.0], [19544258.0, 87591288.0, 144413616.0, 87591288.0, 19544258.0]], dtype=np.float32) 

F3 = np.array([[10.6129, -48.0428, 79.9924, -48.9886, 11.0349], [-48.0428, 217.4419, -361.9798, 221.6424, -49.9172], [79.9924, -361.9798, 602.4865, -368.841, 83.0542], [-48.9886, 221.6424, -368.841, 225.7647, -50.8282], [11.0349, -49.9172, 83.0542, -50.8282, 11.4414]], dtype=np.float32) 

F4 = np.array([[-1.4873, -7.1547, 33.9963, -33.0613, 9.7869], [-7.1547, 92.4115, -244.292, 196.5744, -52.4847], [33.9963, -244.292, 534.3445, -387.8122, 96.4539], [-33.0613, 196.5744, -387.8122, 262.189, -61.6158], [9.7869, -52.4847, 96.4539, -61.6158, 13.7606]], dtype=np.float32) 
'''


local_weights = "weights.png"


def main():
    # Init the class DataManager
    print("===================== load data =========================")
    dataManager = DataManager(img_height, img_width)
    # Get data
    train_data, validation_data = dataManager.get_train_data(
        train_data_dir, validation_data_dir, train_batch_size, val_batch_size)
    # Get class name:id
    label_map = (train_data.class_indices)
    # save model class id
    with open(saved_model_classid_path, 'w') as outfile:
        json.dump(label_map, outfile)
    # Init the class ScratchModel

    model = Model(image_shape, class_number)
    # Get model architecture

    print("===================== load model architecture =========================")
    loaded_model = model.get_model_architecture()
    # plot the model
    plot_model(loaded_model, to_file=model_png)  # not working with windows
    # serialize model to JSON
    model_json = loaded_model.to_json()
    with open(saved_model_arch_path, "w") as json_file:
        json_file.write(model_json)

    # Delete the last summary file
    delete_file(model_summary_file)
    # Add the new model summary
    loaded_model.summary(print_fn=save_summary)
    print("===================== compile model =========================")

    # Compile the model
    loaded_model = model.compile_model(
        loaded_model, model_loss_function, model_optimizer_rmsprop, model_metrics)


    # prepare weights for the model
    Kernels = np.empty([5, 5, 4], dtype=np.float32)
    for i in xrange(0, 5):
        row = np.empty([5, 4], dtype=np.float32)
        for j in xrange(0, 5):
            row[j][0] = KV[i][j]
            row[j][1] = KM[i][j]
            row[j][2] = GH[i][j]
            row[j][3] = GV[i][j]
        Kernels[i] = row

    preprocess_weights = np.reshape(Kernels, (5, 5, 1, 4))

    #loaded_model.summary()
    
    #loaded_model.set_weights([preprocess_weights])

    loaded_model.load_weights(best_weights)

    

    loaded_model = model.compile_model(
        loaded_model, model_loss_function, model_optimizer_rmsprop, model_metrics)


 

    # Prepare callbacks
    csv_log = callbacks.CSVLogger(train_log_path, separator=',', append=False)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    checkpoint = callbacks.ModelCheckpoint(
        train_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(
        log_dir=model_tensorboard_log + "{}".format(time()))
    callbacks_list = [csv_log, tensorboard, checkpoint]

    print("===================== start training model =========================")
    # start training

    history = loaded_model.fit_generator(train_data,
                                  steps_per_epoch=num_of_train_samples // train_batch_size,
                                  epochs=num_of_epoch,
                                  validation_data=validation_data,
                                  validation_steps=num_of_validation_samples // val_batch_size,
                                  verbose=1,
                                  callbacks=callbacks_list)

    print(history)
    print("========================= training process completed! ===========================")
    
if __name__ == "__main__":
    main()
