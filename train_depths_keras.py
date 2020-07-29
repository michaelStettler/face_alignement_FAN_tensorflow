import pandas as pd
import numpy as np
import json
from face_alignment_keras.image_depth_generator import image_depth_generator
from face_alignment_keras.models import depths_network
import tensorflow as tf


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.metrics.append(logs.get('metrics'))


config_file = 'configs/FAN4-3D_depths.json'
# read file
with open(config_file, 'r') as myfile:
    data = myfile.read()

# parse file
config = json.loads(data)

batch_size = config['batch_size']
num_epoch = config['num_epoch']

# load data
df = pd.read_csv(config['csv_file'], dtype=str)
print(df.head())
num_images = len(df.index)

# set multi GPU
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = depths_network(config, num_images)

# print(model.summary())
print("model compiled!")
print()
print("[Config] num_epoch: {}, batch_size: {}".format(num_epoch, batch_size))
print("[Dataset] name: {}".format(config['csv_file']))
print("[Dataset] num_images: {}, total number of iteration: {}".format(num_images, num_images*num_epoch/batch_size))
print("boundaries:", boundaries)

train_generator = image_depth_generator(df, batch_size)
history = LossHistory()
model.fit(train_generator, steps_per_epoch=num_images // batch_size, epochs=num_epoch, use_multiprocessing=True, workers=12, callbacks=[history])

# save models
model.save_weights('weights/' + config['config_name'] + '_weights.tf')

pd.DataFrame.from_dict(history.metrics).to_csv('history/' + config['config_name'] + '_history.csv', index=False)