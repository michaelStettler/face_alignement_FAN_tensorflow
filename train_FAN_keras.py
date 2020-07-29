import pandas as pd
import numpy as np
import json
from tensorflow.keras import backend as K
from face_alignment_keras.models import FAN
from face_alignment_keras.image_generator import get_batch
from face_alignment_keras.image_generator import image_generator
import tensorflow as tf


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


config_file = 'configs/FAN4-3D.json'
# read file
with open(config_file, 'r') as myfile:
    data = myfile.read()

# parse file
config = json.loads(data)

batch_size = config['batch_size']
num_epoch = config['num_epoch']

custom_loss = True

# load data
df = pd.read_csv(config['csv_file'], dtype=str)
print(df.head())
num_images = len(df.index)

train_generator = image_generator(df, num_epoch, batch_size, custom_loss=custom_loss)

# set multi GPU
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    network_size = config['network_size']

    boundaries = []
    for bound in config['schedule']:
        boundaries.append(num_images/batch_size * bound)

    model = FAN(network_size, training=True, custom_loss=custom_loss, lr=config['lr'], boundaries=boundaries)

    if not custom_loss:
        d = 1  # set a value for d ?..
        def normalized_mean_loss(y_true, y_pred):
            l2 = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)))
            return K.mean(l2 / d)
        model.compile(optimizer='rmsprop', loss=normalized_mean_loss, metrics=[normalized_mean_loss])

# print(model.summary())
print("model compiled!")
print()
print("[Config] num_epoch: {}, batch_size: {}".format(num_epoch, batch_size))
print("[Dataset] name: {}".format(config['csv_file']))
print("[Dataset] num_images: {}, total number of iteration: {}".format(num_images, num_images*num_epoch/batch_size))
print("boundaries:", boundaries)
# model.fit([batch_x, batch_y, ds], [batch_y]*network_size, epochs=100, batch_size=batch_size)  # for normal loss
# model.fit([batch_x, batch_y, ds], epochs=100, batch_size=batch_size)  # for custom loss using get_batch
history = LossHistory()
model.fit(train_generator, steps_per_epoch=num_images // batch_size, epochs=num_epoch, use_multiprocessing=True, workers=12, callbacks=[history])  # for custom loss and generator

# save models
# model.save('my_model.tf')
model.save_weights('weights/' + config['config_name'] + '_weights.tf')

pd.DataFrame.from_dict(history.losses).to_csv('history/' + config['config_name'] + '_history.csv', index=False)
