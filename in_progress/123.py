import cv2
import numpy as np
from matplotlib import pyplot as plt
image= cv2.imread('/home/yauhen/temp/1/znak.png')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
_, contours, _= cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours


#gray=img.copy()
#gray,contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,0,0), 0)


for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>700 and w>700:
        continue

    # discard areas that are too small
    if h<200 or w<200:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)


plt.subplot(1,1,1),plt.imshow(image)
plt.title('Regions'), plt.xticks([]), plt.yticks([])
cv2.imwrite("/home/yauhen/temp/1/2/cnt.jpg",image)
plt.show()
