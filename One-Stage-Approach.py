import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree

from PIL import Image
from random import randint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D
from utils import Utils


base_dir = "data/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)
    
if not os.path.exists("./models/"):
    os.mkdir("./models/")
os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))

WORK_DIR = './dataset/'

CLASSES = os.listdir(work_dir)

IMG_SIZE = 224
IMAGE_SIZE = [224, 224]
DIM = (IMG_SIZE, IMG_SIZE)

ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=1024, shuffle=True)

# OverSampling
train_data, train_labels = train_data_gen.next()
sm = SMOTE(sampling_strategy="auto")
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)
train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

# CallBack function
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True
            
my_callback = MyCallback()

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
opt = tf.keras.optimizers.Adam()

# Model
resnet_model = Sequential()
model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False,
                                                   input_shape=(224,224,3),
                                                   pooling='avg',classes=8,
                                                   weights='imagenet')
for layer in model.layers:
    layer.trainable=False
resnet_model.add(model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512,activation='relu'))
resnet_model.add(Dense(8,activation='softmax'))

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), 
           tfa.metrics.F1Score(num_classes=len(CLASSES))]

CALLBACKS = [my_callback]
    
resnet_model.compile(optimizer=opt,
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)

resnet_model.summary()

EPOCHS = 65

# Training
history = resnet_model.fit(train_data_gen,validation_data=(val_data,val_labels), callbacks=CALLBACKS, epochs=EPOCHS)

test_scores = resnet_model.evaluate(test_data, test_labels)
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
pred_labels = resnet_model.predict(test_data)

ut = Utils(history,pred_labels,test_labels,CLASSES)

ut.plot_ROC_curves()

ut.Classification_report()

ut.Confusion_matrix(title="One-Stage Approach")

ut.balanced_accuracy_score()
ut.Mathews_correlation_coefficient()

model_dir = "./models/" + "One_Stage_Approach"
resnet_model.save(model_dir, save_format='h5')
os.listdir(work_dir)

## Load the model
# pretrained_model = tf.keras.models.load_model(model_dir)
# plot_model(pretrained_model, to_file=work_dir + "model_plot.png", show_shapes=True, show_layer_names=True)