# Document Scanner for 8.5x11 sized papers

# Import dependencies
import cv2
import numpy as np

def imshow(img):
    cv2.imshow('demo', img)
    cv2.waitKey(0)
# Read in the input image
img = cv2.imread('hw2.jpg')
imshow(img)

# Gray the image out
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(gray)

# Gaussian Blur?

# Perfom Canny edge detection
edges = cv2.Canny(gray, 75, 200)
imshow(edges)

# Morphological dilation
edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((3, 3)))
imshow(edges)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Search contours for document
for i, cnt in enumerate(contours):
    # Approximate contour polygon
    poly = cv2.approxPolyDP(cnt, 100, True)[:, 0, :]
    # If contour matches document description
    if cv2.contourArea(cnt) > 1000 and len(poly) == 4:
        # Draw detected contour/polygon
        copy = img.copy()
        cv2.drawContours(copy, contours, i , (0, 255, 0), -1)
        for pt in poly:
            cv2.circle(copy, tuple(pt), 20, (255, 255, 255), -1)
        poly = np.float32(poly)
        imshow(copy)
        # Transform perspective 
        M = cv2.getPerspectiveTransform(poly, np.float32([[850, 0], [850, 1100], [0, 1100], [0,0]]))
        # Apply perspective transform
        dst = cv2.warpPerspective(img, M, (850, 1100))

# Display final result
imshow(dst)