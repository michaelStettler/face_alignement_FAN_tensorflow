import cv2
import numpy as np
import math


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
manual,klo0
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)

    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(int)

# def crop(im, bbox, scale=1.15, verbose=True):
#     """
#     own crop function but I'm using now the same as the paper
#     """
#
#     img_height, img_width, channels = im.shape
#     x0, y0, x1, y1, score = bbox
#     x0 = int(round(x0))
#     x1 = int(round(x1))
#     y0 = int(round(y0))
#     y1 = int(round(y1))
#
#     # get width/height
#     width = x1 - x0
#     height = y1 - y0
#
#     # get center og images
#     c_x = x0 + width / 2
#     c_y = y0 + height / 2
#
#     # control that the height or width size doesn't go out of the picture
#     width_ = scale * width
#     height_ = scale * height
#     if width_ > img_width:
#         width_ = img_width
#     if height_ > img_height:
#         height_ = img_height
#
#     # get max factor to ensure a square size picture
#     max_factor = int(np.maximum(width_, height_))
#
#     # compute back the bbox from the center
#     x0_ = int(round(c_x - max_factor / 2))
#     x1_ = x0_ + max_factor
#     y0_ = int(round(c_y - max_factor / 2))
#     y1_ = y0_ + max_factor
#
#     # chek that the bbox does not go outside of the img frame
#     if x0_ < 0:
#         diff = - x0_
#         x0_ = 0
#         x1_ = x1_ + diff
#
#     if x1_ > img_width - 1:
#         diff = x1_ - img_width
#         x0_ = x0_ - diff
#         x1_ = img_width - 1
#
#     if y0_ < 0:
#         diff = - y0_
#         y0_ = 0
#         y1_ = y1_ + diff
#
#     if y1_ > img_height - 1:
#         diff = y1_ - img_width
#         y0_ = y0_ - diff
#         y1_ = img_height - 1
#
#     if verbose:
#         print("img_width", img_width, "img_height:", img_height)
#         print("max_factor:", max_factor)
#         print("x0: ", x0_, "x1:", x1_, " y0:", y0_," y1:", y1_)
#
#     return x0_, y0_, x1_, y1_


def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    idx = np.argmax(np.reshape(hm, (hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])), 2)
    idx += 1
    preds = np.reshape(idx, (idx.shape[0], idx.shape[1], 1)).repeat(2, 2).astype(float)
    preds[..., 0] = np.apply_along_axis(lambda x: (x - 1) % hm.shape[3] + 1, 1, preds[..., 0])
    preds[..., 1] = np.floor((preds[..., 1] - 1) / hm.shape[2]) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * .25

    preds -= .5

    preds_orig = np.zeros(preds.shape)
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.shape[2], True)

    return preds, preds_orig


def create_target_heatmap(target_landmarks, centers, scales):
    heatmaps = np.zeros((target_landmarks.shape[0], 68, 64, 64), dtype=np.float32)
    for i in range(heatmaps.shape[0]):
        for p in range(68):
            landmark_cropped_coor = transform(target_landmarks[i, p] + 1, centers[i], scales[i], 64, invert=False)
            heatmaps[i, p] = draw_gaussian(heatmaps[i, p], landmark_cropped_coor + 1, 1)
    return heatmaps