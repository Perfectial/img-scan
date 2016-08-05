import os
import numpy as np
import cv2
import time
import sys
from sklearn.cluster import DBSCAN
from itertools import groupby
from operator import itemgetter

MIN_RADIUS = 12

DEBUG = False
DEMO_SLEEP = 0
if 'debug' in sys.argv:
    DEBUG = True
    DEMO_SLEEP = 0.04

IMAGE_PATH = 'static/test_sample.jpg'
# IMAGE_PATH = '../samples/pdf_scan_marks.jpg'
if len(sys.argv) > 1:
    filename = sys.argv[-1]
    if filename != 'debug':
        IMAGE_PATH = filename

#IMAGE_PATH = '../samples/pdf_scan_marks.jpg'
#IMAGE_PATH = '../samples/dummy_scan_1.png'


if not os.path.exists(IMAGE_PATH):
    print('File does not exist: %s' % IMAGE_PATH)

image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
clean_image = np.copy(image)
cimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_circles(circles):
    c_image = np.copy(image)
    for i in circles[0, :]:
        cv2.circle(c_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(c_image, (i[0], i[1]), 1, (0, 0, 255), 3)
        if DEBUG:
            cv2.imshow('output', c_image)
            cv2.waitKey(1)
        time.sleep(DEMO_SLEEP / 20.)
    cv2.imshow('output', c_image)
    cv2.waitKey(1)
    time.sleep(DEMO_SLEEP*20)


print('Draft scan for all circles...\n')
circles = cv2.HoughCircles(cimg,
                           cv2.HOUGH_GRADIENT,
                           dp=0.5,
                           minDist=5,
                           param1=2,
                           param2=50,
                           minRadius=5,
                           maxRadius=int(cimg.shape[1]/20))

sorted_radius = np.sort(circles[:, :, -1])
median = int(np.median(sorted_radius, axis=1)[-1])
stdev = np.std(sorted_radius, axis=1)

circles = np.uint16(np.around(circles))
if DEBUG:
    draw_circles(circles)

print('Precise analysis with thresholds tweaking...\n')

circles = cv2.HoughCircles(cimg,
                           cv2.HOUGH_GRADIENT,
                           dp=0.5,
                           minDist=median,
                           param1=2,
                           param2=20,
                           minRadius=median-max(1, stdev),
                           maxRadius=median+max(1, stdev))

sorted_radius = np.sort(circles[:, :, -1])
median = int(np.median(sorted_radius, axis=1)[-1])
stdev = int(np.std(sorted_radius, axis=1)[0])

circles = np.uint16(np.around(circles))
if DEBUG:
    draw_circles(circles)

print('Final analysis with accurate thresholds...\n')

circles = cv2.HoughCircles(cimg,
                           cv2.HOUGH_GRADIENT,
                           dp=0.5,
                           minDist=median,
                           param1=2,
                           param2=30,
                           minRadius=median-max(1, stdev),
                           maxRadius=median+max(1, stdev))

circles = np.uint16(np.around(circles))
draw_circles(circles)

sorted_radius = np.sort(circles[:, :, -1])
median = int(np.median(sorted_radius, axis=1)[-1])
MIN_RADIUS = median+1
stdev = int(np.std(sorted_radius, axis=1)[0])

print('Detecting cluster of bubbles according to their coordinates...')
circles_xy = circles[0, :, :-1]
db = DBSCAN(eps=2*2*median).fit(circles_xy)
labels = db.labels_
n_clusters = len(set(labels))
labeled_clusters = np.insert(circles_xy, 2, labels, axis=1)
labeled_clusters = labeled_clusters[np.argsort(labeled_clusters[:, -1])]

clusters = []
ignore_num = 0
for k, vals in groupby(labeled_clusters, key=itemgetter(2)):
    if k in (-1, 65535):
        ignore_num += 1
        continue
    points = np.array(list(vals))[:, :-1]
    clusters.append(points)
clusters = np.array(clusters)
print('Found %s/%s valid clusters (ignored %s cluster)\n' % (len(clusters), n_clusters, ignore_num))

print('Finding cluster boundaries with further arrangement...\n')
max_points = [cluster.max(axis=0) for cluster in clusters]
min_points = [cluster.min(axis=0) for cluster in clusters]
sorted_x = [np.sort(cluster[:, 0]) for cluster in clusters]
sorted_y = [np.sort(cluster[:, 1]) for cluster in clusters]
split_by_x = [np.split(x, np.where(np.diff(x) > 10)[0]+1) for x in sorted_x]
split_by_y = [np.split(y, np.where(np.diff(y) > 10)[0]+1) for y in sorted_y]
n_choices = [[len(x)] for x in np.array(split_by_x)]
n_questions = [[len(y)] for y in np.array(split_by_y)]

cluster_specs = np.hstack((max_points, min_points, n_choices, n_questions))
cluster_specs = cluster_specs[cluster_specs[:, 1].argsort()]

tests = np.split(cluster_specs, np.where(np.diff(cluster_specs[:, 1]) > 0)[0]+1)
test_clusters = []
for clusters in tests:
    clusters = np.sort(clusters, axis=0)
    test_clusters.append([np.dstack(np.meshgrid(np.sort(np.linspace(x[0], x[2], num=x[-2])), np.sort(np.linspace(x[1], x[3], num=x[-1])))) for x in clusters])

print('Split all clusters on %s test groups...\n' % len(test_clusters))

clu_image = np.copy(clean_image)
for min_x, min_y, max_x, max_y, _, _ in cluster_specs:
    cv2.rectangle(clu_image, (min_x+MIN_RADIUS, min_y+MIN_RADIUS), (max_x-MIN_RADIUS, max_y-MIN_RADIUS), (230, 0, 0), 3)
    cv2.imshow('output', clu_image)
    cv2.waitKey(1)
    time.sleep(DEMO_SLEEP)

cv2.imshow('output', clu_image)
cv2.waitKey(1)
time.sleep(DEMO_SLEEP*10)


def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


ocr_image = np.copy(clean_image)

print('Analyzing all choices for marks detection...')

for test_id, cluster_circles in enumerate(test_clusters):
    questions_offset = 0
    for clu_id, questions in enumerate(cluster_circles):
        for q_id, choices in enumerate(questions):
            q_sums = []
            for c_id, choice in enumerate(choices):
                x, y = choice
                cimg = clean_image[y-MIN_RADIUS:y+MIN_RADIUS, x-MIN_RADIUS:x+MIN_RADIUS]
                q_sums.append(cimg.sum())
                cv2.rectangle(clu_image, (int(x-MIN_RADIUS), int(y-MIN_RADIUS)), (int(x+MIN_RADIUS), int(y+MIN_RADIUS)), (0, 100, 220))
                if DEBUG:
                    cv2.imshow('output', clu_image)
                    cv2.waitKey(1)
                time.sleep(DEMO_SLEEP/2)

            mark_results = is_outlier(np.array(q_sums))
            mark_idx = np.argwhere(mark_results == True)
            if len(mark_idx) > 0:
                x, y = choices[mark_idx[0]][0]
                print("[Test %s] Question %s: -> choice[%s]\t\t %s === %s" % (test_id+1, q_id+1+questions_offset, mark_idx[0][0]+1, q_sums, mark_results))

                cimg = np.copy(clean_image[y-MIN_RADIUS:y+MIN_RADIUS, x-MIN_RADIUS:x+MIN_RADIUS])
                cv2.rectangle(clu_image, (int(x-MIN_RADIUS), int(y-MIN_RADIUS)), (int(x+MIN_RADIUS), int(y+MIN_RADIUS)), (0, 200, 0), 3)
                x1, y1 = choices[0]
                x1 -= 70
                y1 -= 10
                clu_image[y1:y1+cimg.shape[1], x1:x1+cimg.shape[0]] = cimg
        questions_offset += q_id + 1

cv2.imshow('output', clu_image)
cv2.waitKey(0 if DEBUG else 1)
