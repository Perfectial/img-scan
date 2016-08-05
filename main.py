import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from itertools import groupby
from operator import itemgetter


IMAGE_PATH = 'static/test_sample.jpg'
MIN_RADIUS = 11

if not os.path.exists(IMAGE_PATH):
    print('File does not exist: %s' % IMAGE_PATH)

image = cv2.imread(IMAGE_PATH, 0)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(image,
                           cv2.HOUGH_GRADIENT,
                           dp=0.5,
                           minDist=7,
                           param1=2,
                           param2=15,
                           minRadius=MIN_RADIUS,
                           maxRadius=15)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 1, (0, 0, 255), 3)

# cv2.imshow('output', cimg)
# cv2.waitKey(0)

circles_xy = circles[0, :, :-1]
db = DBSCAN(eps=50).fit(circles_xy)
labels = db.labels_
n_clusters_ = len(set(labels))
labeled_clusters = np.insert(circles_xy, 2, labels, axis=1)
labeled_clusters = labeled_clusters[np.argsort(labeled_clusters[:, -1])]

clusters = []
for k, vals in groupby(labeled_clusters, key=itemgetter(2)):
    if k in (-1, 65535):
        continue
    points = np.array(list(vals))[:, :-1]
    clusters.append(points)
clusters = np.array(clusters)

max_points = [cluster.max(axis=0) for cluster in clusters]
min_points = [cluster.min(axis=0) for cluster in clusters]
sorted_x = [np.sort(cluster[:, 0]) for cluster in clusters]
sorted_y = [np.sort(cluster[:, 1]) for cluster in clusters]
split_by_x = [np.split(x, np.where(np.diff(x) > 10)[0]+1) for x in sorted_x]
split_by_y = [np.split(y, np.where(np.diff(y) > 10)[0]+1) for y in sorted_y]
n_choices = [[len(x)] for x in np.array(split_by_x)]
n_questions = [[len(y)] for y in np.array(split_by_y)]

cluster_specs = np.hstack((max_points, min_points, n_choices, n_questions))
cluster_specs = cluster_specs[np.lexsort((cluster_specs[:, 0], cluster_specs[:, 1]))]


circles_by_cluster = [np.dstack(np.meshgrid(np.sort(np.linspace(x[0], x[2], num=x[-2])), np.sort(np.linspace(x[1], x[3], num=x[-1])))) for x in cluster_specs]


def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# for clust_id, questions in enumerate(circles_by_cluster):
#     for q_id, choices in enumerate(questions):
#         q_sums = []
#         for c_id, choice in enumerate(choices):
#             print('>>> Processing cluster %s', clust_id)
#             x, y = choice
#             cimg = image[y-MIN_RADIUS:y+MIN_RADIUS, x-MIN_RADIUS:x+MIN_RADIUS]
#             q_sums.append(cimg.sum())
#             cv2.rectangle(image, (int(x-MIN_RADIUS), int(y-MIN_RADIUS)), (int(x+MIN_RADIUS), int(y+MIN_RADIUS)), (0, 0, 250))
#             cv2.imshow('output', image)
#             cv2.waitKey(0)
#
#         outlier = is_outlier(np.array(q_sums))

min_coordinates = [x[0] for x in circles_by_cluster]
matrix_min_coordinates = []
for i in np.array(min_coordinates):
    for j in i:
        matrix_min_coordinates.append(j)
section_min_points = np.split(matrix_min_coordinates, np.where(np.diff(np.array(matrix_min_coordinates)[:, -1]) > 10)[0] + 1)

max_coordinates = [x[-1] for x in circles_by_cluster]
matrix_max_coordinates = []
for i in np.array(max_coordinates):
    for j in i:
        matrix_max_coordinates.append(j)
section_max_points = np.split(matrix_max_coordinates, np.where(np.diff(np.array(matrix_max_coordinates)[:, -1]) > 10)[0] + 1)


for min_p, max_p in zip(section_min_points, section_max_points):
    cv2.rectangle(image, (int(min_p[:, 0][0])-MIN_RADIUS, int(min_p[:, 1][-1])-MIN_RADIUS),
                  (int(max_p[:, 0][-1])+MIN_RADIUS, int(max_p[:, 1][-1])+MIN_RADIUS),
                  (0, 0, 255), 3)
cv2.imshow('output', image)
cv2.waitKey(0)
