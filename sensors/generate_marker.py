import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker = cv2.aruco.generateImageMarker(
    dictionary, 0, 400
)

# Add white border for reliable detection
marker = cv2.copyMakeBorder(
    marker, 40, 40, 40, 40,
    cv2.BORDER_CONSTANT,
    value=255
)

cv2.imwrite("assets/aruco_marker_0.png", marker)

print("SUCCESS: ArUco marker generated")
