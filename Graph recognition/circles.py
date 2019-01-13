import numpy as np
import cv2
import math
import sys

def is_circle(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    radius = int(radius)
    circle_area_rate = 2
    area = cv2.contourArea(cnt)
    return radius*radius*math.pi < area*circle_area_rate

def point_distance(p1, p2):
    p1 = p1[0]
    p2 = p2[0]
    return math.sqrt(((p1[0]-p2[0])*(p1[0]-p2[0]))+((p1[1]-p2[1])*(p1[1]-p2[1])))

def get_distance(cnt1, cnt2):
    max_dist = 10000000
    for p1 in cnt1:
        for p2 in cnt2:
            dist = point_distance(p1, p2)
            max_dist = min(max_dist, dist)
    return max_dist

def get_nearest_circles(circles, line):
    circle_distances = [get_distance(c, line) for c in circles]
    sorted_list = sorted(circle_distances)
    return (circle_distances.index(sorted_list[0]), circle_distances.index(sorted_list[1]))

image = cv2.imread(sys.argv[1])

#avg_color = np.average(image)

#cv2.threshold(image,avg_color,255,cv2.THRESH_BINARY)

(height, width, _) = image.shape

output = np.array(image)
output.fill(255)
area_rate = 10000
image_area = height*width
expected_area = image_area/area_rate
contour_thickness = 2
lower = np.array([0, 0, 0])
upper = np.array([1, 1, 1])

# detect circles in the image
shapeMask = cv2.inRange(image, lower, upper)


# find the contours in the mask
(_,contours,_) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lines = []
circles = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > expected_area:
        if is_circle(cnt):
            circles.append(cnt)
        else:
            lines.append(cnt)

for line in lines:
    c1, c2 = get_nearest_circles(circles, line)
    print(str(c1) + " - " + str(c2))

cv2.drawContours(output, lines, -1, (0,255,0), thickness=contour_thickness)
cv2.drawContours(output, circles, -1, (0,0,255), thickness=contour_thickness)

cv2.imwrite('pics/contours.png', output)
