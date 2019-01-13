window_size = 5
img_width = 600
img_height = 600


noise_ratio = 0.0001
threshold_block_size_ratio = 0.09
threshold_mean = 14


threshold=0.4
min_sigma=8

#convolution size for searching centroids
#higher value - less tolerance, use no less than 2 and no more than 3
line_width_ratio = 2.5

#how much can be white
acceptable_rate=0.02

# lower: less edges, higher: more edges
shi_tomasi_threshold_mean = -30
shi_tomasi_threshold_block_size = 151
edge_acceptance_in_blob = 0.09

max_edges_distance = img_width*0.12

edge_blob_distance_limit = img_width*0.15

#values used to measure edge direction, we leave last spare_edge pixels, beause end of edge is noisy
# how much edge are we leaving
spare_edge_ratio = 0.05
measure_edge = 10

minimum_edge_length = 20

PICS_FOLDER = 'pics/'

PREPROC_RESULT = PICS_FOLDER+'result.png'
NO_CORNERS = PICS_FOLDER+'no_corners.png'
NO_CORNERS_THIN = PICS_FOLDER+'no_corners_thin.png'
DEBUG_IMAGE = PICS_FOLDER+'debug.png'

COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_PURPLE = (255,0,255)
