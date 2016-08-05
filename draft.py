import numpy as np
import cv2


img = np.zeros((512, 512, 3), np.uint8)
# draw a line
# cv2.line(img, (15, 20), (70, 60), (255, 0, 0), 5)

# draw a circle
# cv2.circle(img, (200, 200), 40, (0, 0, 255), -1)

# draw rectangle
cv2.rectangle(img, (15, 20), (70, 50), (0, 0, 255), 3)

# pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.polylines(img, [pts], True, (0, 255, 255))
#
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Hello', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('KKK', img)
cv2.waitKey(0)