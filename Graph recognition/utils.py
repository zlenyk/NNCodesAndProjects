import cv2
from PIL import Image
from Queue import *
import numpy as np
from scipy import stats, signal
from sets import Set
import math, matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import time
import config
import line_iterator

def open_image(path):
    pil_image = Image.open(path).resize((config.img_height, config.img_width), Image.ANTIALIAS).convert('L')
    orig_img = np.array(pil_image)
    return orig_img

def show_image(label, img, gap=0):
    if isinstance(img, Image.Image):
        img = np.array(img)
    #resized = cv2.resize(img, (800,800), interpolation = cv2.INTER_AREA)
    resized = img
    #cv2.namedWindow(label,cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(label, 800,800)

    cv2.imshow(label,resized)
    cv2.waitKey(gap*1000)
    cv2.destroyWindow(label)

def distance(p1, p2):
    return int(math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])))

def normalize(array, upper_limit):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array *= upper_limit
    return array

# Applies convolution of kernel ones()
# looks for squares with acceptance ratio
def convolve_image(convolve_size, image):
    radius = int(convolve_size/2)
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1

    binary_image = (np.copy(image)).astype(float)
    binary_image[binary_image > 0] = 1
    max_acceptable = (convolve_size*convolve_size) * config.acceptable_rate
    conv = signal.convolve(binary_image, kernel, mode='same')
    conv[conv <= max_acceptable] = 0
    conv[conv > max_acceptable] = 255
    return conv

# convolves image with large filter (2*line_width), what is left is a vertex
# connected components create vertices
def vertices_centroids(image, line_width):
    conv = convolve_image(convolve_size=config.line_width_ratio*line_width, image = image)
    I,J = np.where(conv==0)
    visited = Set()
    component = {}
    current_comp = 0
    for i in range(len(I)):
        if (I[i], J[i]) in visited:
            continue
        q = Queue()
        q.put((I[i], J[i]))
        while not q.empty():
            top = q.get()
            if top in visited:
                continue
            visited.add(top)
            if top[0] >= 0 and top[0] < config.img_width and top[1] >= 0 and top[1] < config.img_height \
                    and conv[top] == 0:
                component[top] = current_comp
                visited.add(top)
                q.put((top[0]-1, top[1]))
                q.put((top[0]-1, top[1]-1))
                q.put((top[0]-1, top[1]+1))
                q.put((top[0]+1, top[1]-1))
                q.put((top[0]+1, top[1]+1))
                q.put((top[0]+1, top[1]))
                q.put((top[0], top[1]-1))
                q.put((top[0], top[1]+1))
        current_comp += 1

    centroids = np.zeros((current_comp, 2))
    comp_size = np.zeros((current_comp))
    for indice in component:
        centroids[component[indice]] += [indice[0], indice[1]]
        comp_size[component[indice]] += 1

    shape_diff = np.subtract(image.shape, conv.shape)
    centroids /= comp_size[:, None]
    return centroids, shape_diff, conv

def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in xrange(0,n+1)]

def get_smallest_neighbour(point, image, centroid, rgb_img, min_val=100000):
    min_ind = (-2,-2)
    l = image[int(point[0])-1:int(point[0])+2:,int(point[1])-1:int(point[1])+2].flatten()
    dist = distance(point, centroid)
    for i in range(-1,2):
        for j in range(-1,2):
            neigh = [point[0]+i,point[1]+j]
            if distance(neigh, centroid) >= dist and \
                    not np.array_equal(rgb_img[int(neigh[0])][int(neigh[1])], [0,255,0]) and \
                    image[int(neigh[0])][int(neigh[1])] < min_val:
                min_ind = (i,j)
                min_val = image[int(neigh[0])][int(neigh[1])]
    return point + min_ind

#return blob indice, and +/- 1 (-1 if end of edge is nearer than beg)
# end: 0 for beg, 1 for end
def closest_blob_for_edge(edge, blobs, end):
    closest_dist = config.img_height
    closest_ind = -1
    edge_end = edge[0]
    if end == 1:
        edge_end = edge[-1]
    for i in range(len(blobs)):
        y, x, _ = blobs[i]
        if distance([y,x], edge_end) < closest_dist:
            closest_dist = distance([y,x], edge_end)
            closest_ind = i
    return blobs[closest_ind]

def vector_distance(vec1, vec2):
    return (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2

#return closest edge indice and flipped: +/- {1,2}
# -2, both flipped, -1 our flipped, 1 other flipped, 2 none flipped
def closest_edge(edge, edges):
    closest_dist = config.img_height
    closest_ind = -1
    flipped = 0
    edge_beg, edge_end = edge[0], edge[-1]

    for i in range(len(blobs)):
        y, x, _ = blobs[i]
        if distance([y,x], edge_beg) < closest_dist:
            closest_dist = distance([y,x], edge_beg)
            closest_ind = i
            flipped = 1
        if distance([y,x], edge_end) < closest_dist:
            closest_dist = distance([y,x], edge_beg)
            closest_ind = i
            flipped = -1
    return closest_ind, flipped

#end : 0 - beg, 1 - end, 2 - any
#return 0 - beg, 1 - end
def edge_match_blob(edge, blob, blobs, end=2):
    if end == 0 or end == 2:
        closest_blob = closest_blob_for_edge(edge, blobs, 0)
        if np.array_equal(closest_blob, blob):
            y, x, _ = closest_blob
            if distance([y, x], edge[0]) < config.edge_blob_distance_limit:
                edge_blob_distance = distance([y, x], edge[0])
                #edge_vector has always length ~ 10
                edge_vector = edge_end_vector(edge, 0, 0)
                new_point = edge[0] + (edge_blob_distance/10)*edge_vector
                if distance([y, x], new_point) < edge_blob_distance:
                    return True, 0
    if end == 1 or end == 2:
        closest_blob = closest_blob_for_edge(edge, blobs, 1)

        if np.array_equal(closest_blob, blob):
            y, x, _ = closest_blob
            if distance([y, x], edge[-1]) < config.edge_blob_distance_limit:
                edge_blob_distance = distance([y, x], edge[-1])
                #edge_vector has always length ~ 10
                edge_vector = edge_end_vector(edge, 1, 0)
                new_point = edge[-1] + (edge_blob_distance/10)*edge_vector

                if distance([y, x], new_point) < edge_blob_distance:
                    return True, 1
    return False, 0

# if edge matches more than 1 blob (e.g connects 2 blobs) either one will be returned
# return -1 if None found
def find_edge_match_blob(edge, blobs, end=2):
    for i in range(len(blobs)):
        result, flip = edge_match_blob(edge, blobs[i], blobs, end)
        if result:
            return i, flip
    return -1, 0

def edge_spare_end(edge):
    return int(max(10, len(edge)*config.spare_edge_ratio))

#direction: 0 - outside edge, 1 - inside edge
def edge_end_vector(edge, end, direction):
    result = None
    edge = np.array(edge)
    if end == 0:
        result = edge[edge_spare_end(edge)] - edge[edge_spare_end(edge) + config.measure_edge]
    else:
        result = edge[len(edge) - edge_spare_end(edge)] - edge[len(edge) - edge_spare_end(edge) - config.measure_edge]
    #now is measured always outside edge, if want inside, we have to multiply by -1
    if direction == 1:
        return -result
    return result

# checks if edge1 and edge2 match, returns True/False, flip - 0/1,
# 0 - beginning of edge2, 1 - end
# param: end - 0, beg of edge1, 1 - end
def edge_match_edge(edge1, edge2, end):
    end_vector_edge1 = edge_end_vector(edge1, end, 0)
    end_vector_edge2 = None
    edge2_end = 0
    if min(distance(edge1[end*(len(edge1)-1)], edge2[0]), distance(edge1[end*(len(edge1)-1)], edge2[len(edge2)-1]) ) > config.max_edges_distance:
        return 1000, edge2_end
    if distance(edge1[end*(len(edge1)-1)], edge2[0]) < distance(edge1[end*(len(edge1)-1)], edge2[len(edge2)-1]):
        # we go from beg of edge1 to beg of edge2
        end_vector_edge2 = edge_end_vector(edge2, 0, 1)
        edge2_end = 0
    else:
        # from beg of edge1 to end of edge2
        end_vector_edge2 = edge_end_vector(edge2, 1, 1)
        edge2_end = 1
    return np.sum(np.square(end_vector_edge1 - end_vector_edge2)), edge2_end


#removes corners areas from edges
#harris_result - result of harris function (255 - corners, 0 - edges)
def remove_corners(image, blobs, edges):
    edges_no_corners = np.copy(image)
    edges_no_vertices = np.copy(image)
    edges_no_corners[edges == 255] = 255
    for blob in blobs:
        y, x, r = blob
        new_r  = r+15
        for i in range(int(x-new_r), int(x+new_r+1)):
            for j in range(int(y-new_r), int(y+new_r+1)):
                if (i-x)*(i-x) + (j-y)*(j-y) < new_r:
                    edges_no_corners[j,i] = 255
                    edges_no_vertices[j,i] = 255
    return edges_no_corners, edges_no_vertices

# returns list of edges (each edge is list of points)
# image - thin image of edges (1 - edge, 0 - no edge)
def get_edges_list(image, edges_ends):
    edges_no_corners = np.copy(image)
    edges_no_corners[edges_ends == 1] = 2
    visited = np.zeros_like(edges_no_corners)
    edges = []
    for p in np.argwhere(edges_no_corners == 2):
        edge = []
        if visited[p[0],p[1]] == 0:
            while True:
                visited[p[0],p[1]] = 1
                edge.append([p[0], p[1]])
                new_p = p
                # we do it in strange way (these phases), to check adjacent first (not on cross)
                found = False
                for i in (-1,1):
                    if visited[p[0]+i,p[1]] == 0 and edges_no_corners[p[0]+i,p[1]] > 0:
                        new_p = np.array([p[0]+i,p[1]])
                        found = True
                        break
                    if visited[p[0],p[1]+i] == 0 and edges_no_corners[p[0],p[1]+i] > 0:
                        new_p = np.array([p[0],p[1]+i])
                        found = True
                        break
                if not found:
                    for i in (-1,1):
                        for j in (-1,1):
                            if visited[p[0]+i,p[1]+j] == 0 and edges_no_corners[p[0]+i,p[1]+j] > 0:
                                new_p = np.array([p[0]+i,p[1]+j])
                                break

                if np.array_equal(new_p,p):
                    break
                p = new_p
            if len(edge) > config.minimum_edge_length:
                edges.append(edge)

    return edges

def edges_connecting_edges(edges, blobs):
    edges_used = []
    connected_blobs = []
    point_point_connection = []
    end_lambda = lambda x, edge: edge_spare_end(edge) if x==0 else -edge_spare_end(edge)
    for i in range(len(edges)):
        if i in edges_used:
            continue
        edges_used.append(i)

        current_blob_index, flip = find_edge_match_blob(edges[i], blobs)
        if current_blob_index == -1:
            continue
        current_edge = edges[i]

        y, x, r = blobs[current_blob_index]
        point_point_connection.append((current_edge[end_lambda(flip, current_edge)], [y,x]))

        while True:
            end_blob_index, end_flip = find_edge_match_blob(current_edge, blobs, end=abs(1-flip))
            if end_blob_index > -1:
                connected_blobs.append([current_blob_index,end_blob_index])
                y, x, r = blobs[end_blob_index]
                point_point_connection.append((current_edge[end_lambda(1-flip, current_edge)],[y,x]))
                break

            # [(result, end),...], return result 1000 if edge is not good
            results_ends = [edge_match_edge(current_edge, edges[j], abs(1-flip)) for j in range(len(edges))]

            results_list = sorted((min_result, index, edge2_end) for (index, (min_result, edge2_end)) in enumerate(results_ends) if index not in edges_used and min_result < 1000)
            if len(results_list) == 0:
                break

            (min_result, index, edge2_end) = results_list[0]
            point_point_connection.append((current_edge[end_lambda(1-flip, current_edge)], edges[index][end_lambda(edge2_end, edges[index])]))
            current_edge = edges[index]
            flip = edge2_end
            edges_used.append(index)

    return connected_blobs, point_point_connection


def are_neighbours(p1, p2):
    return abs(p1[0][0] - p2[0][0]) <= 1 and abs(p1[0][1] - p2[0][1]) <= 1

def is_white(point, img):
    return img[int(point[0])][int(point[1])] == 255

def point_in_list(plist, point):
    for p in plist:
        if np.array_equal(point, p):
            return True
    return False

def neighs_in_list(nlist, neighs):
    for n in nlist:
        if (np.array_equal(n[0], neighs[0]) and np.array_equal(n[1], neighs[1])) \
            or (np.array_equal(n[1], neighs[0]) and np.array_equal(n[0], neighs[1])):
            return True
    return False

# returns point en edge 5 points after current
def get_neighbour(point, img):
    neigh_list=[]
    prev = None
    while(len(neigh_list) < config.window_size):
        got_cand = False
        for i in range(-1, 2):
            for j in range(-1, 2):
                if got_cand:
                    break
                neigh_candidate = np.copy(point)
                neigh_candidate[0] += i
                neigh_candidate[1] += j
                if neigh_candidate[0] >= config.img_height or neigh_candidate[0] < 0 or \
                    neigh_candidate[1] > config.img_width or neigh_candidate[0] < 0:
                    continue
                if is_white(neigh_candidate, img) \
                    and not point_in_list(neigh_list, neigh_candidate):
                    neigh_list.append(neigh_candidate)
                    prev = point
                    point = neigh_candidate
                    got_cand = True
        if not got_cand:
            return None
    if len(neigh_list) == config.window_size:
        return (neigh_list[0], neigh_list[4])

# search closes white pixel from p1 in direction of p2
def closest_white_between(beg, end, img):
    min_dist = 1000
    line_iter = line_iterator.createLineIterator(beg, end, img)
    for point in line_iter[1:]:
        if is_white(point, img):
            min_dist = min(min_dist, distance(beg, point))
    return min_dist

def search_closest_white(n1, n2, img):
    center = np.asarray([int(round((n1[0]+n2[0])/2)), int(round((n1[1]+n2[1])/2))])
    vec = n1 - n2
    # perpendicular
    vec = [vec[1], -vec[0]]
    c1 = center + np.multiply(vec,10)
    return min(closest_white_between(center, center + np.multiply(vec,5), img),
                    closest_white_between(center, center - np.multiply(vec,5), img))

def neigh_tuple(neigh):
    return (neigh[0][0],neigh[0][1],neigh[1][0], neigh[1][1])

# returns array with distances between colses edges on image
def edge_distances(image):
    # returns white edges and black everything else
    edge_img = cv2.Canny(image, 100, 200)
    cv2.imwrite('pics/edges.png', edge_img)

    # get all edge points (white)
    points = np.argwhere(edge_img > 0)
    neigh_set = Set()
    distances = []
    for point in points:
        neigh = get_neighbour(point, edge_img)
        # this neigh_in_list is slow
        if neigh == None or neigh_tuple(neigh) in neigh_set:
            continue

        neigh_set.add(neigh_tuple(neigh))

        center = np.asarray([int(round((neigh[0][0]+neigh[1][0])/2)), int(round((neigh[0][1]+neigh[1][1])/2))])
        min_dist = search_closest_white(neigh[0], neigh[1], edge_img)
        distances.append(min_dist)
    return distances

#remove unnecesarry centroids (for example too close)
def check_blobs(blobs):
    new_centroids = []
    for blob in blobs:
        y, x, r = blob
        blocking_centroid = [(_y,_x,_r) for _y,_x,_r in new_centroids if distance([y,x], [_y, _x]) < 30]
        if len(blocking_centroid) == 0:
            new_centroids.append(blob)
    return new_centroids

def get_line_width(image):
    distances = edge_distances(image)
    unique, counts = np.unique(distances, return_counts=True)

    #plt.plot(unique[:18], counts[:18])
    #plt.ylabel('ilosc wystapien')
    #plt.xlabel('grubosc')

    #plt.savefig('distances.png', bbox_inches='tight')
    count_inds = np.array(counts).argsort()[::-1][:len(counts)]
    sorted_values = unique[count_inds]

    line_width = sorted_values[0]
    if line_width == 1000:
        line_width = sorted_values[1]
    return line_width

def check_blobs_against_old_centroids(blobs, image, line_width):
    centroids, shape_diff, conv = vertices_centroids(image, line_width)

    conv = conv.astype('uint8')
    conv = cv2.cvtColor(conv,cv2.COLOR_GRAY2RGB)
    conv = draw_circles(blobs, conv, config.COLOR_RED, 2)
    cv2.imwrite('pics/conv.png', conv)

    new_blobs = []
    for blob in blobs:
        x,y,r = blob
        for centroid in centroids:
            xc,yc = centroid
            if distance([x,y], [xc,yc]) < 30:
                new_blobs.append(blob)
                break
    return new_blobs

#R - harris funcion, 0 - edge, R_8 - unthresholded harris function
# returns new blobs, points in edges
def check_blobs_against_edges(blobs, image, line_width):
    orig_reversed = abs(255-image)

    eigen = cv2.cornerEigenValsAndVecs(orig_reversed, blockSize=2*line_width+3, ksize=3)

    k = 0.1
    R = eigen[:,:,0]*eigen[:,:,1] - k * np.square(eigen[:,:,0]+eigen[:,:,1])
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    R *= 255
    R = abs(R - 255)
    cv2.imwrite('pics/shi_tomasi_orig.png', R)

    #check different blures, this seems by far the best
    blur = cv2.blur(orig_reversed,(3,3))

    eigen = cv2.cornerEigenValsAndVecs(blur, blockSize=2*line_width+3, ksize=3)

    k = 0.1
    R = eigen[:,:,0]*eigen[:,:,1] - k * np.square(eigen[:,:,0]+eigen[:,:,1])
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    R *= 255
    R = abs(R - 255)
    cv2.imwrite('pics/shi_tomasi.png', R)

    R_8 = R.astype('uint8')

    R = cv2.adaptiveThreshold(
        R_8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=config.shi_tomasi_threshold_block_size,
        C=config.shi_tomasi_threshold_mean,
    )
    cv2.imwrite('pics/shi_threshold.png', R)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb_img[R == 255] = (255,0,0)
    cv2.imwrite('pics/rgb_shi.png', rgb_img)
    new_blobs = []
    R = R.astype(int)
    for blob in blobs:
        bx,by,r = blob
        bx,by = int(bx), int(by)
        a, b = 1, 1
        r = int(r-1)
        n = int(2*r+1)
        y,x = np.ogrid[-r:r+1, -r:r+1]
        mask = x*x + y*y <= r*r
        kernel = np.zeros((n, n))
        kernel[mask] = 1
        conv = signal.convolve(R, kernel, mode='same')
        if conv[bx,by] < 255 * int(3.14*r*r) * config.edge_acceptance_in_blob:
            new_blobs.append(blob)
    R = abs(255-R)
    return new_blobs, R

#return array of size as input array where 1 mean that black pixel has at most 1 neighbour
def count_neighbours(input_array):
    kernel = np.array([[1,1,1],
                        [1,10,1],
                        [1,1,1]])
    conv = signal.convolve(input_array, kernel, mode='same')
    neighs = np.zeros_like(input_array)
    neighs[(conv>=10)&(conv<=11)] = 1
    return neighs

def draw_circles(blobs, rgb_img, color=(255,0,0), width=-1):
    for blob in blobs:
        y, x, r = blob
        cv2.circle(rgb_img, (int(x), int(y)), int(r), color, 2)
    return rgb_img
