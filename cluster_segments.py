from colorizer import segment_image, get_pixels_for_segment
from color_space_converter import *
from math import fabs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from texture_extraction import calculate_centroid_from_segment
from sklearn.cluster import KMeans
import math
def get_average_luminance_per_segment(img, number_of_segments):
    segments = segment_image(img, number_of_segments)
    pixels_per_segment = get_pixels_for_segment(segments)
    centroids = list(map(calculate_centroid_from_segment, pixels_per_segment))
    avg_lum = np.zeros(len(pixels_per_segment))
    img_lab = rgb_to_yuv(img)
    i = 0
    for segment in pixels_per_segment:
        for (x, y) in segment:
            avg_lum[i] += img_lab[x, y][0]

        avg_lum[i] /= len(segment)
        i += 1
    return avg_lum, segments, centroids


def get_segment_distance_matrix(avg_luminance, neighbours):
    dist_matrix = np.zeros((len(avg_luminance), len(avg_luminance)))
    for i in range(len(avg_luminance)):
        for j in range(len(avg_luminance)):
            factor = 1
            if j not in neighbours[i]:
                dist_matrix[i, j] = 1000000
            else:
                dist_matrix[i, j] = fabs(avg_luminance[i] - avg_luminance[j])
    return dist_matrix

def get_neighbouring_segments(segments):
    number_of_segments = np.max(segments) + 1
    neighbours = [set() for _ in range(number_of_segments)]

    THRESHOLD = 50
    for (x, y), value in np.ndenumerate(segments):
        if y < len(segments[x]) - 1:
            adj_value = segments[x][y + 1]
            if adj_value != value :
                neighbours[value].add(adj_value)
                neighbours[adj_value].add(value)

        if x < len(segments) - 1:
            adj_value = segments[x + 1][y]
            if adj_value != value :
                neighbours[value].add(adj_value)
                neighbours[adj_value].add(value)

    return neighbours


def get_segments_by_label(label, segments, labels):
    selected_indices = []
    for i in range(len(labels)):
        if labels[i] != label:
            selected_indices.append(i)
    segments_some = np.copy(segments)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            if segments_some[i, j] in selected_indices:
                segments_some[i, j] = -1
    # sel = np.array(selected_indices)
    # segments_some[sel[:, None], :] = -1
    return segments_some


img = read_img("/home/LinuxData/colorizer/recolorizer/data/test/39373580184.jpg")
avg_lum, segments, centroids = get_average_luminance_per_segment(img, 500)
neighbours = get_neighbouring_segments(segments)
dist_matrix = get_segment_distance_matrix(avg_lum, neighbours)

clusterizer = DBSCAN(metric="precomputed")
labels = clusterizer.fit_predict(dist_matrix)
img_seg = mark_boundaries(img, get_segments_by_label(6, segments, labels))
plt.imshow(img_seg)
plt.show()
print("yea")