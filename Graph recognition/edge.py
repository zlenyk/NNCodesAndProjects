import cv2
from thinner import Thinner
import utils, config
from skimage.feature import blob_log
import time
import argparse
import numpy as np
import graph_draw
import argparse
import preproc_utils

def operation_wrapper(name, function, args):
    start = time.time()
    result = function(**args)
    end = time.time()
    print(name," time: ", end-start)
    return result

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=preproc_utils.str2bool, help="Test mode")

args = parser.parse_args()

test = args.t
if test == None:
    test = False

orig_img = utils.open_image(config.PREPROC_RESULT)

line_width = operation_wrapper("Get line width", utils.get_line_width, {'image':orig_img})

blobs = operation_wrapper("Blob detector", blob_log, {'image': abs(255-orig_img), 'threshold':config.threshold, 'min_sigma':line_width*1.5})
first_blobs = blobs
if test == True:
    debug_picture, _ = graph_draw.debug_picture(orig_img, first_blobs=first_blobs)
    utils.show_image("Blob detection", np.array(debug_picture), 0)

blobs = operation_wrapper("Check blobs against old centroids", utils.check_blobs_against_old_centroids, {'blobs':blobs, 'image':orig_img, 'line_width':line_width})
second_blobs = blobs
if test == True:
    debug_picture, _ = graph_draw.debug_picture(orig_img, first_blobs=first_blobs, second_blobs=second_blobs)
    utils.show_image("Circular kernel", np.array(debug_picture), 0)

blobs, edges = operation_wrapper("Check blobs against edges", utils.check_blobs_against_edges, {'blobs':blobs, 'image':orig_img, 'line_width':line_width})
third_blobs = blobs
if test == True:
    debug_picture, _ = graph_draw.debug_picture(orig_img, first_blobs=first_blobs, second_blobs=second_blobs, third_blobs=third_blobs)
    utils.show_image("Blobs on edges", np.array(debug_picture), 0)

blobs = operation_wrapper("Remove too close blobs", utils.check_blobs, {'blobs':blobs})
fourth_blobs = blobs
if test == True:
    debug_picture, _ = graph_draw.debug_picture(orig_img, first_blobs=first_blobs, second_blobs=second_blobs, third_blobs=third_blobs, fourth_blobs=fourth_blobs)
    utils.show_image("Too close blobs", np.array(debug_picture), 0)

edges_no_corners, edges_no_vertices = operation_wrapper("Remove corners", utils.remove_corners, {'image':orig_img, 'blobs':blobs, 'edges':edges})
cv2.imwrite(config.NO_CORNERS, edges_no_corners)
if test == True:
    utils.show_image("Remove corners", np.array(edges_no_corners), 0)

edges_no_corners_thin = operation_wrapper("Thin edges", Thinner.thin_image, {'image':edges_no_corners})
cv2.imwrite(config.NO_CORNERS_THIN, edges_no_corners_thin)
if test == True:
    utils.show_image("Thin edges", np.array(edges_no_corners_thin), 0)

edges_no_corners_thin[edges_no_corners_thin == 0] = 1
edges_no_corners_thin[edges_no_corners_thin == 255] = 0
edges_ends = utils.count_neighbours(edges_no_corners_thin)

edges = operation_wrapper("Get edges list", utils.get_edges_list, {"image":edges_no_corners_thin, "edges_ends":edges_ends})

graph, point_point_connection = operation_wrapper("Edge matching edges", utils.edges_connecting_edges, {'edges':edges, 'blobs':blobs})

#graph = [list(x) for x in set(tuple(x) for x in connected_blobs)]
print("Line width:", line_width)
print(graph)

debug_picture, edges_no_vertices = graph_draw.debug_picture(orig_img, edges_no_corners=edges_no_corners, edges_no_vertices=edges_no_vertices, edges=edges, blobs=blobs, line_width=line_width, point_point_connection=point_point_connection, first_blobs=first_blobs, second_blobs=second_blobs, third_blobs=third_blobs, fourth_blobs=fourth_blobs)
cv2.imwrite(config.DEBUG_IMAGE, debug_picture)
cv2.imwrite('pics/edges_no_vertices.png', edges_no_vertices)

graph_draw.draw_graph(blobs, graph, 'pics/dot')

print("Blobs: " + str(len(fourth_blobs)))
print("Edges: " + str(len(graph)))

end = time.time()
print "Time: ", end-start
