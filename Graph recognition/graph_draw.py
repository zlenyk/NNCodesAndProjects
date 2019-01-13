from graphviz import Graph
import numpy as np
import cv2
import utils, config

def draw_graph(blobs, graph, filename):
    dot = Graph(engine='neato')
    for i in range(len(blobs)):
        y, x, r = blobs[i]
        y /= -15
        x /= 15
        dot.node(str(i), pos=str(x)+","+str(y)+"!", penwidth="5")
    for [a,b] in graph:
        dot.edge(str(a), str(b), penwidth="5")
    dot.render(filename=filename)

def debug_picture(orig_img, edges_no_corners=None, edges_no_vertices=None, edges=None, blobs=None, line_width=None, point_point_connection=None, first_blobs=None, second_blobs=None, third_blobs=None, fourth_blobs=None):
    debug_image = np.copy(orig_img)
    debug_image = cv2.cvtColor(debug_image,cv2.COLOR_GRAY2RGB)

    if not edges_no_vertices is None:
        edges_no_vertices = cv2.cvtColor(edges_no_vertices,cv2.COLOR_GRAY2RGB)

    if not edges_no_corners is None:
        debug_image[edges_no_corners==0] = (0,255,255)

    if not (line_width is None or point_point_connection is None):
        for (point1, point2) in point_point_connection:
            point1 = np.flip([int(x) for x in point1], 0)
            point2 = np.flip([int(x) for x in point2], 0)
            cv2.line(debug_image, tuple(point1), tuple(point2), (0,220,220), line_width)
    if not first_blobs is None:
        debug_image = utils.draw_circles(first_blobs, debug_image, config.COLOR_RED)
    if not second_blobs is None:
        debug_image = utils.draw_circles(second_blobs, debug_image, config.COLOR_PURPLE)
    if not third_blobs is None:
        debug_image = utils.draw_circles(third_blobs, debug_image, config.COLOR_BLUE)
    if not fourth_blobs is None:
        debug_image = utils.draw_circles(fourth_blobs, debug_image, config.COLOR_GREEN)
        if not edges_no_vertices is None:
            edges_no_vertices = utils.draw_circles(fourth_blobs, edges_no_vertices, config.COLOR_GREEN, 3)

    return debug_image, edges_no_vertices
