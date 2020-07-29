import numpy as np
import json
import tensorflow as tf
import pandas as pd
import cv2
from face_alignment_keras.models import FAN
from face_alignment_keras.image_generator import get_batch
from face_alignment_keras.utils import get_preds_fromhm
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_file = 'configs/predict_FAN4-3D.json'
# read file
with open(config_file, 'r') as myfile:
    data = myfile.read()

# parse file
config = json.loads(data)

batch_size = config['batch_size']
num_epoch = 1

# load images
df = pd.read_csv(config['csv_file'], dtype=str)
print(df.head())
print("num_images", len(df.index))

# load model
model = FAN(config['network_size'], custom_loss=True, lr=config['lr'], boundaries=config['schedule'])
model.load_weights('weights/' + config['config_name'] + '_weights.tf')
# print(model.summary())

# predict heatmaps
batch_x, batch_y, ds = get_batch(df, shuffle=False)
heatmaps = model.predict([batch_x, batch_y, ds])
print("shape heatmaps", np.shape(heatmaps))
print(np.ndim(heatmaps))
if np.ndim(heatmaps) > 4:
    heatmaps = heatmaps[-1]
print("shape heatmaps", np.shape(heatmaps))
heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))
print("shape heatmaps", np.shape(heatmaps))

# predict 2d positions from heatmaps
preds = get_preds_fromhm(heatmaps)[0]
print("shape preds", np.shape(preds))

inp = np.zeros((64, 64, 3))
# control preds
for i in range(68):
    x = preds[0, i, 0]
    y = preds[0, i, 1]

    inp[int(y), int(x), :] = [255, 255, 255]

# save image
cv2.imwrite('test/'+ config['config_name'] +'_hm_preds.jpg', cv2.cvtColor(inp.astype('uint8'), cv2.COLOR_RGB2BGR))


