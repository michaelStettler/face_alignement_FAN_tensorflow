import os
import pandas as pd
import cv2
import numpy as np
from face_alignment_keras.utils import *

"""
Control the predictions using the face-alignment pipeline!
"""

# declare variables
# df = pd.read_csv('preds.csv')
df = pd.read_csv('preds_img.csv')
print(df.head())

is_3d = True

for index, row in df.iterrows():
    im_name = row['img']
    print("image name:", im_name)
    im = cv2.imread(im_name)[..., ::-1]  # read it as rgb
    print("image size:", im.shape)

    # control cropping
    # get the detected faces
    cx = int(row['cx'])
    cy = int(row['cy'])
    scale = int(row['scale'])
    center = [cx, cy]
    inp = crop(im, center, scale)

    # save image
    cv2.imwrite(im_name[:-4] + '_cropped.jpg', cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

    # control landmarks
    for i in range(68):
        x = row[str(i) + '_x']
        y = row[str(i) + '_y']

        inp[int(y), int(x), :] = [0, 255, 0]

    # save image
    cv2.imwrite(im_name[:-4] + '_lmk.jpg', cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

    # create heatmaps
    landmarks = row.to_numpy()
    print("shape landmarks", np.shape(landmarks))
    target_landmarks = landmarks[5:]  # keep only the landmarks pos
    if is_3d:
        target_landmarks = np.reshape(target_landmarks, (-1, 3))
        target_landmarks = target_landmarks[:, 0:2]
    else:
        target_landmarks = np.reshape(target_landmarks, (-1, 2))
    target_landmarks = np.reshape(target_landmarks, (-1, 2))
    target_landmarks = np.expand_dims(target_landmarks, 0)  # add the batch dim
    scales = np.expand_dims(scale, 0)  # add the batch dim
    centers = np.expand_dims(center, 0)  # add the batch dim
    print("shape target_landmarks", np.shape(target_landmarks))
    heatmaps = create_target_heatmap(target_landmarks, [[128, 128]], [1])
    heatmaps = np.array(heatmaps)
    print("shape heatmaps", np.shape(heatmaps))

    # save image
    heatmap_0 = heatmaps[0, 0]
    print("shape heatmap_0", np.shape(heatmap_0))
    cv2.imwrite(im_name[:-4] + '_heatmap_0.jpg', heatmap_0*255)

    heatmaps = np.sum(heatmaps, 1)
    print("shape heatmaps", n
    p.shape(heatmaps))
    cv2.imwrite(im_name[:-4] + '_heatmaps.jpg', heatmaps[0]*255)
