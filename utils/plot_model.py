import numpy as np
import tensorflow as tf
import pandas as pd
import json
from face_alignment_keras.models import FAN
import matplotlib.pyplot as plt
from face_alignment_keras.image_generator import get_batch
from face_alignment_keras.utils import get_preds_fromhm

config_file = 'configs/catA.json'
# read file
with open(config_file, 'r') as myfile:
    data = myfile.read()

# parse file
config = json.loads(data)

# load history
history = pd.read_csv('history/'+config['config_name']+'_history.csv')
history = history.to_numpy()

# create plot
plt.plot(history)
plt.savefig('figures/'+config['config_name']+'_figures.png')
