import numpy as np
import cv2
from face_alignment_keras.utils import get_preds_fromhm

hm = np.load('test_hm.npy')
true_preds = np.load('test_preds.npy')
# print("shape hm", np.shape(hm))
# print(hm[0, :, 0])
# print("shape true preds", np.shape(true_preds))
# print(true_preds[0, :, 0])

preds = get_preds_fromhm(hm)[0]

inp = np.zeros((64, 64, 3))
# control preds
for i in range(68):
    x = preds[0, i, 0]
    y = preds[0, i, 1]

    inp[int(y), int(x), :] = [255, 255, 255]

# save image
cv2.imwrite('test/test_keras_hm_preds.jpg', cv2.cvtColor(inp.astype('uint8'), cv2.COLOR_RGB2BGR))