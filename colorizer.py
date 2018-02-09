from matplotlib.colors import Colormap
from skimage.segmentation import slic
from skimage.data import imread
from skimage.util import img_as_float, img_as_ubyte
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

from color_space_converter import *
from sklearn.svm import SVR
import os
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from color_smoothing import get_neighbouring_segments, smooth_colors
from sklearn.linear_model import Ridge, Lasso
from texture_extraction import *
from sklearn.neural_network import MLPRegressor

PATH = "/home/thalvadzhiev/Documents/{0}"
path_man_mirror = PATH.format("nature2.jpg")


def segment_image(image, number_of_segments):
    return slic(image, n_segments=number_of_segments, compactness=10, sigma=1)


def get_pixels_for_segment(segments):
    """gets coordinates of pixels per segment"""
    pixels_for_segment = [[] for _ in range(segments.max() + 1)]
    for (x, y), segment_label in np.ndenumerate(segments):
        pixels_for_segment[segment_label].append((x, y))
    return np.array([np.array(segment) for segment in pixels_for_segment])


def sample_segment_with(pixels_for_segment, sample_size):
    """sample 'sample_size' points randomly from every segment"""
    segments_sampled = [[] for _ in range(len(pixels_for_segment))]
    for segment_index, segment in enumerate(pixels_for_segment):
        segments_sampled[segment_index] = segment[np.random.choice(len(segment), sample_size)]
    return np.array(segments_sampled)


def get_lum_chrom_triple_sample(yuv_image, sampled_segments):
    """get a luminance vector for every segment and a vector with average chrominance per channel per segment"""

    luminance = np.zeros((sampled_segments.shape[0], sampled_segments.shape[1]))
    u = np.zeros(sampled_segments.shape[0])
    v = np.zeros(sampled_segments.shape[0])
    for index, segment in enumerate(sampled_segments):
        for index_in_segment, (x, y) in enumerate(segment):
            luminance[index][index_in_segment] = yuv_image[x, y][0]
            u[index] += yuv_image[x, y][1]
            v[index] += yuv_image[x, y][2]
        luminance[index] = np.fft.fft(luminance[index])
    return luminance, u / sampled_segments.shape[1], v / sampled_segments.shape[1]


def get_histogram_per_segment(pixels_per_segment, yuv_img, n_bins):
    """get normalized luminance histogram per segment"""
    histograms = []
    for segment in pixels_per_segment:
        lum_segment = []
        for (x, y) in segment:
            lum_segment.append(yuv_img[x, y][0])
        hist, _ = np.histogram(lum_segment, bins=n_bins, range=(0, 1))
        hist = hist / hist.sum()
        histograms.append(hist)
    return np.array(histograms)

def get_descriptors_per_segment(img, pixels_per_segment):
    descriptors = []
    for segment in pixels_per_segment:
        descs = get_descriptors_from_segment(np.array(img), 20, segment)
        if descs is not None:
            descriptors.append(descs.flatten())
        else:
            descriptors.append(np.zeros(64))
    return np.asarray(descriptors)


def get_lum_chrom_triple_image(path, number_of_segments, sample_size):
    img = read_img(path)
    segments = segment_image(img, number_of_segments)
    pixels_per_segment = get_pixels_for_segment(segments)
    sampled_segments = sample_segment_with(pixels_per_segment, sample_size)
    yuv = rgb_to_yuv(img_as_float(img))
    return get_lum_chrom_triple_sample(yuv, sampled_segments)

def get_lum_chrom_hist(path, number_of_segments, sample_size, n_bins):

    img = read_img(path)
    segments = segment_image(img, number_of_segments)
    pixels_per_segment = get_pixels_for_segment(segments)
    descriptors = get_descriptors_per_segment(img, pixels_per_segment)
    sampled_segments = sample_segment_with(pixels_per_segment, sample_size)
    lab = rgb_to_yuv(img_as_float(img))
    lum, a, b = get_lum_chrom_triple_sample(lab, sampled_segments)
    hist = get_histogram_per_segment(pixels_per_segment, lab, n_bins)
    lum_hist = np.hstack((lum, hist))
    lum_hist_desc = np.hstack((lum_hist, descriptors))
    return lum_hist_desc, a, b

def predict_image(path, u_predictor, v_predictor, number_of_segments, sample_size, n_bins, pca, smoothing_limit):
    img = read_img(path)
    segments = segment_image(img, number_of_segments)
    pixels_per_segment = get_pixels_for_segment(segments)
    descriptors = get_descriptors_per_segment(img, pixels_per_segment)

    sampled_segments = sample_segment_with(pixels_per_segment, sample_size)
    yuv_img = rgb_to_yuv(img_as_float(img))
    lum, u, v = get_lum_chrom_triple_sample(yuv_img, sampled_segments)
    hist = get_histogram_per_segment(sampled_segments, yuv_img, n_bins)
    lum_hist = np.hstack((lum, hist))
    lum_hist_desc = np.hstack((lum_hist, descriptors))
    x_reduced = pca.transform(lum_hist_desc)
    u_predicted = clamp_u(u_predictor.predict(x_reduced) * 3)
    v_predicted = clamp_v(v_predictor.predict(x_reduced) * 3)
    u_smoothed, v_smoothed = smooth_colors(u_predicted, v_predicted, segments, lum, smoothing_limit)

    return img, color_picture(yuv_img, u_smoothed, v_smoothed, pixels_per_segment)

def predict_image_yuv(path, a_predictor, b_predictor, number_of_segments, sample_size, n_bins, pca):
    color_map = (yuv_to_rgb, rgb_to_yuv)
    return predict_image(path, a_predictor, b_predictor, number_of_segments, sample_size,  n_bins, pca)


def color_picture(yuv_img_float, u_predicted, v_predicted, pixels_per_segment):
    img = np.copy(yuv_img_float)
    for segment_index, segment in enumerate(pixels_per_segment):
        u = u_predicted[segment_index]
        v = v_predicted[segment_index]
        for x, y in segment:
            img[x, y][1] = u
            img[x, y][2] = v
    return yuv_to_rgb(img)


def generate_all_feature_vectors(path, number_of_segments, sample_size, n_bins):
    """path is the path to the directory containing train images"""
    u_all = np.array([])
    v_all = np.array([])
    lum_all = np.array([]).reshape((0, sample_size + n_bins + 64))
    i = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg"):
                i += 1
                print("\rWorking on picture {0} of {1}".format(i, len(files)), end="")
                lum, u, v = get_lum_chrom_hist(path + "/" + filename, number_of_segments,
                                               sample_size, n_bins)
                u_all = np.concatenate((u_all, u))
                v_all = np.concatenate((v_all, v))
                lum_all = np.concatenate((lum_all, lum))
    return lum_all, u_all, v_all


def train_svr(path, model_name, number_of_segments, sample_size, n_bins, retrain=False):
    if retrain:
        LUM, A, B = generate_all_feature_vectors(path, number_of_segments, sample_size, n_bins)
        a_svr = SVR(C=0.1, epsilon=0.03)
        b_svr = SVR(C=0.1, epsilon=0.03)
        pca = PCA(n_components=100)
        print("\nFitting PCA..")
        pca.fit(LUM)
        print("Transforming with PCA..")
        LUM = pca.transform(LUM)
        print("Fitting model 1...")
        a_svr.fit(LUM, A)
        print("Fitting model 2...")
        b_svr.fit(LUM, B)
        print("Done!")
        joblib.dump(a_svr, "models/cr/svr_a_" + model_name)
        joblib.dump(b_svr, "models/br/svr_b_" + model_name)
        joblib.dump(pca, "models/pca/pca_" + model_name)

    else:
        a_svr = joblib.load("models/cr/svr_a_" + model_name)
        b_svr = joblib.load("models/br/svr_b_" + model_name)
        pca = joblib.load("models/pca/pca_" + model_name)

    return a_svr, b_svr, pca


def train_ridge_lasso(path, model_name, number_of_segments, sample_size, color_map, retrain=False):
    if retrain:
        LUM, A, B = generate_all_feature_vectors(path, number_of_segments, sample_size, color_map)
        a_svr = Ridge()
        b_svr = Ridge()

        print("\nFitting model...")
        a_svr.fit(LUM, A)
        b_svr.fit(LUM, B)
        print("Done!")
        joblib.dump(a_svr, "models/cr/ridge_a_" + model_name)
        joblib.dump(b_svr, "models/br/ridge_b_" + model_name)
    else:
        a_svr = joblib.load("models/cr/ridge_a_" + model_name)
        b_svr = joblib.load("models/br/ridge_b_" + model_name)

    return a_svr, b_svr


def train_mlp(path, model_name, number_of_segments, sample_size, color_map, n_bins, retrain=False):
    if retrain:
        LUM, A, B = generate_all_feature_vectors(path, number_of_segments, sample_size, color_map, n_bins)
        a_svr = MLPRegressor(hidden_layer_sizes=200, activation="logistic", alpha=0.1)
        b_svr = MLPRegressor(hidden_layer_sizes=200, activation="logistic", alpha=0.1)

        print("\nFitting model 1...")
        a_svr.fit(LUM, A)
        print("Done with model 1!")
        print("Fitting model 2...")
        b_svr.fit(LUM, B)
        print("Done!")
        joblib.dump(a_svr, "models/cr/mlp_a_" + model_name)
        joblib.dump(b_svr, "models/br/mlp_b_" + model_name)
    else:
        a_svr = joblib.load("models/cr/mlp_a_" + model_name)
        b_svr = joblib.load("models/br/mlp_b_" + model_name)

    return a_svr, b_svr


def main():
    number_of_segments = 500
    sample_size = 300
    n_bins = 50

    cr_svr, br_svr, pca = train_svr("/home/LinuxData/colorizer/recolorizer/data/train", "svr_landscapes_pca_500", number_of_segments,
                               sample_size, n_bins, False)
    img, predicted = predict_image("/home/LinuxData/colorizer/recolorizer/data/test/39373580184.jpg",
                                       cr_svr, br_svr, number_of_segments, sample_size, n_bins, pca, 50)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(predicted)
    plt.show()


if __name__ == "__main__":
    main()
