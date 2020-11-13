import cv2

img = cv2.imread('sudoku_puzzles/7.jpg')
img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)

cv2.rectangle(img, (100,100),(200,200), (255, 0, 0), 1)
print(img.shape)
cv2.imshow('sup', img)
cv2.waitKey(0)
