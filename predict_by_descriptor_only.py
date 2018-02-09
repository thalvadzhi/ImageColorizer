from sklearn.externals import joblib

from colorizer import get_descriptors_per_segment, segment_image, get_pixels_for_segment, get_lum_chrom_triple_image, color_picture
from color_space_converter import *
import os
from sklearn.svm import SVR
import matplotlib.pyplot as plt
def get_desc(path):
    img = read_img(path)
    segments = segment_image(img, 500)
    pixels_per_segment = get_pixels_for_segment(segments)
    descs= get_descriptors_per_segment(img, pixels_per_segment)
    lum, a, b = get_lum_chrom_triple_image(path, 500, 300)
    return descs, a, b

def predict_image(path, svr_a, svr_b):
    img = read_img(path)
    segments = segment_image(img, 500)
    pixels_per_segment = get_pixels_for_segment(segments)
    descriptors = get_descriptors_per_segment(img, pixels_per_segment)
    lab_img = rgb_to_yuv(img_as_float(img))
    a_predicted = clamp_u(svr_a.predict(descriptors) * 3)
    b_predicted = clamp_v(svr_b.predict(descriptors) * 3)

    return img, color_picture(lab_img, a_predicted, b_predicted, pixels_per_segment)

def train(path, retrain=False):
    if retrain:
        i = 0
        descs = np.array([]).reshape((0, 64))
        a_s = np.array([])
        b_s = np.array([])
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(".jpg"):
                    i += 1
                    print("\rWorking on picture {0} of {1}".format(i, len(files)), end="")
                    desc, a, b = get_desc(path + "/" + filename)
                    descs = np.concatenate((descs, desc), axis=0)
                    a_s = np.concatenate((a_s, a), axis=0)
                    b_s = np.concatenate((b_s, b), axis=0)
        svr_a = SVR()
        svr_b = SVR()
        svr_a.fit(descs, a_s)
        svr_b.fit(descs, b_s)
        joblib.dump(svr_a, "models/test/svr_a_" + "kot_takoa")
        joblib.dump(svr_b, "models/test/svr_b_" + "kot_takoa")
    else:
        svr_a = joblib.load("models/test/svr_a_" + "kot_takoa")
        svr_b = joblib.load("models/test/svr_b_" + "kot_takoa")
    return svr_a, svr_b

a, b= train("/home/LinuxData/colorizer/recolorizer/data/train", False)
img, img_pred= predict_image("/home/LinuxData/colorizer/recolorizer/data/test/40084181501.jpg", a, b)
f, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(img_pred)
plt.show()
