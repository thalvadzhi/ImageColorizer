import cv2
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=5000, upright=True)


def get_descriptor_around_centroid(img, centroid, square_side):
    #must be an rgb image
    centroid_x, centroid_y = round(centroid[0]), round(centroid[1])
    delta = square_side // 2
    top_row = int(max(centroid_y - delta, 0))
    bottom_row = int(min(centroid_y + delta, img.shape[0]))
    left_column = int(max(centroid_x - delta, 0))
    right_column = int(min(centroid_x + delta, img.shape[1]))
    img_window = img[top_row:bottom_row, left_column:right_column]
    key_point = cv2.KeyPoint(x=centroid_x, y=centroid_y, _size=delta)
    _, descs = surf.compute(img_window, keypoints=[key_point], descriptors=None)
    return descs


def calculate_centroid_from_segment(pixels_in_segment):
    sum_x = sum(map(lambda point: point[0], pixels_in_segment))
    sum_y = sum(map(lambda point: point[1], pixels_in_segment))
    return sum_x / len(pixels_in_segment), sum_y / len(pixels_in_segment)


def get_descriptors_from_segment(img, square_side, pixels_in_segment):
    centroid = calculate_centroid_from_segment(pixels_in_segment)
    return get_descriptor_around_centroid(img, centroid, square_side)
