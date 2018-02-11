from PIL import Image
from skimage.util import img_as_float
import numpy as np
from constants import *


YUV_TO_RGB = np.array([[1, 0, 1.13983],
                       [1, -0.39465, -.58060],
                       [1, 2.03211, 0]]).T
RGB_TO_YUV = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]]).T


def read_img(path):
    return Image.open(path)


def read_img_as_float(path):
    return img_as_float(Image.open(path))


def clamp(value, lower_bound, upper_bound):
    upper_clamp = np.minimum(value, upper_bound)
    return np.maximum(upper_clamp, lower_bound)


def clamp_u(val):
    return clamp(val, U_MIN, U_MAX)


def clamp_v(val):
    return clamp(val, V_MIN, V_MAX)


def clamp_cr(val):
    return clamp(val, -Cr_MAX, Cr_MAX)


def clamp_br(val):
    return clamp(val, -Br_MAX, Br_MAX)


def yuv_to_rgb(img):
    return clamp(np.dot(img, YUV_TO_RGB), 0, 1)


def rgb_to_yuv(img):
    return np.dot(img, RGB_TO_YUV)


def rgb_to_ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 0.5
    return ycbcr


def ycbcr_to_rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.344136, -.714136], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 0.5
    rgg = rgb.dot(xform.T)
    rgg[rgg < 0] = 0
    rgg[rgg > 1] = 1
    return rgg


def rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
