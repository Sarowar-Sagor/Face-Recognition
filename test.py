import cv2
image=cv2.imread('SAGOR.jpg')
img=cv2.resize(image,(600,600))
cv2.imshow('image',img)
cv2.waitKey(0)