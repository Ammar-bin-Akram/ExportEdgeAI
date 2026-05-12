import cv2

name = 'snapshot_healthy.jpg'
frame = cv2.imread(name)

# Define ROI (x, y, width, height)
x, y, w, h = 174, 55, 342, 447
roi = frame[y:y+h, x:x+w]

new_name = 'snapshot_healthy_cropped.jpg'
cv2.imshow("Cropped ROI", roi)
cv2.imwrite(new_name, roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
