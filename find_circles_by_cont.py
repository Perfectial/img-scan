import cv2
# import imutils
import numpy as np


img = cv2.imread("static/file1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(),
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]
for c in cnts:
    M = cv2.moments(c)
    peri = cv2.arcLength(c, True)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print '>>', M, cX, cY
    cv2.drawContours(img, [c], -1, (0, 255, 0), 1)
    cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
cv2.imshow('output', img)
cv2.waitKey(0)
