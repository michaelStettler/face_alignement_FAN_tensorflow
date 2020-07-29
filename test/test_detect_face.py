from face_alignment_keras.detection.s3fd.s3fd_detector import S3FD
import cv2
import numpy as np
import os
from face_alignment_keras.utils import crop
print("directory:", os.listdir())


def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im


# Get the face detector
face_detector = S3FD()

# img_path = '../../../FaceAlignment/LS3D-W/300VW-3D/CatA/547_/0001.jpg'
img_path = '../../../FaceAlignment/LS3D-W/300W-Testset-3D/indoor_044.png'
# img_path = '../../../FaceAlignment/LS3D-W/300W-Testset-3D/indoor_051.png'
# img_path = '../../../FaceAlignment/LS3D-W/300W-Testset-3D/indoor_068.png'

im = cv2.imread(img_path)[..., ::-1]  # switch to rgb
im_height, im_width, channels = im.shape
cv2.imwrite("test/original_image.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

# get the detected faces on original image
bboxes = face_detector.detect_face(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
x0, y0, x1, y1, score = bboxes[0]  # show the first detected face
x0_ = int(round(x0))
x1_ = int(round(x1))
y0_ = int(round(y0))
y1_ = int(round(y1))
print("x0: ", x0_, "x1:", x1_, " y0:", y0_," y1:", y1_)

# draw bbox
bbox = im
bbox[y0_:y1_, x0_, :] = [0, 0, 255]
bbox[y0_:y1_, x1_, :] = [0, 0, 255]
bbox[y0_, x0_:x1_, :] = [0, 0, 255]
bbox[y1_, x0_:x1_, :] = [0, 0, 255]
cv2.imwrite("test/face_bbox.jpg", cv2.cvtColor(bbox, cv2.COLOR_RGB2BGR))

# width = x1_ - x0_
# height = y1_ - y0_
# print("width:", width, " height:", height)
# center = [x0_ + int(round((x1_ - x0_)/2)), y0_ + int(round((y1_ - y0_)/2))]
# print("center: ", center)
#
# scale = im_height / height
# print("im height ", im_height, "face height ", height, "scale ", scale)
#
# face = crop(im, center, 2*scale)
# cv2.imwrite("test/croped.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

# # resize image
# im = resize_image(im)  # Resize image to prevent GPU OOM.
# cv2.imwrite("test/resized_image.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
# img_height, img_width, channels = im.shape

# # get the detected faces on resized image
# bboxes = face_detector.detect_face(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
# x0, y0, x1, y1, score = bboxes[0]  # show the first detected face
# x0_ = int(round(x0))
# x1_ = int(round(x1))
# y0_ = int(round(y0))
# y1_ = int(round(y1))
# print("x0: ", x0_, "x1:", x1_, " y0:", y0_," y1:", y1_)
# width = x1_ - x0_
# height = y1_ - y0_
# print("width:", width, " height:", height)
# face = im[y0_:y1_, x0_:x1_, :]
# cv2.imwrite("test/face_image.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

x0_, y0_, x1_, y1_ = crop(im, bboxes[0])
cropped_face = im[y0_:y1_, x0_:x1_, :]
cv2.imwrite("test/croped_face_image.jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

