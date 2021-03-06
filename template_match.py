import numpy as np
import cv2

img = cv2.imread('static/file1.jpg')
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray = cv2.imread('static/file1.jpg',0)

ret,thresh = cv2.threshold(gray,127,255,1)

_, contours, h = cv2.findContours(thresh,1,2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    # print len(approx)
    # if len(approx)==5:
    #     print "pentagon"
    #     cv2.drawContours(img,[cnt],0,255,-1)
    # elif len(approx)==3:
    #     print "triangle"
    #     cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    # elif len(approx)==4:
    #     print "square"
    #     cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    # elif len(approx) == 9:
    #     print "half-circle"
    #     cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    if len(approx) > 5:
        print "circle"
        cv2.drawContours(img,[cnt],0,(0,255,255),-1)

cv2.imshow('img',gray)
cv2.waitKey(0)