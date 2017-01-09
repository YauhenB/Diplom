import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/yauhen/temp/1/с2.jpg')


imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,120,120,120)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#gray=img.copy()
#gray,contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,0,0), 0)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in contours[0]]
#contours,boxes=sort_contours(contours)
idx = 0 

for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    if h>1000 and w>1000:
        continue

    # discard areas that are too small
    if h<13 or w<13:
        continue

    roi=img[y:y+h,x:x+w]
    cv2.imwrite('/home/yauhen/temp/1/2/'+str(idx) + '.jpg', roi)
    cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
    
cv2.polylines(img, hulls, 1, (0, 0, 0), 3)
plt.subplot(1,1,1),plt.imshow(img)
plt.title('Regions'), plt.xticks([]), plt.yticks([])

plt.show()


