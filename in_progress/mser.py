import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/yauhen/temp/1/znak.png',0)

mser = cv2.MSER_create(5,700,5000,0.25,0.2,200,1.05,0.003,10)
gray_img = img.copy()

regions = mser.detectRegions(gray_img, None)

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(img)
 
blimage = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(gray_img, hulls, 1, (0, 0, 0), 2)

print(hulls)

plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(gray_img)
plt.title('Regions'), plt.xticks([]), plt.yticks([])

plt.show()
