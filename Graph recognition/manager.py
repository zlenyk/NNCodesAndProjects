import sys
import argparse
import glob
import subprocess
import preproc_utils, config
import time
import numpy as np
import re
import networkx as NX
from networkx.algorithms import isomorphism

parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, type=int, help='first file')
parser.add_argument('-l', required=False, type=int, help='last file')
parser.add_argument('-t', required=False, type=preproc_utils.str2bool, help='test')
parser.add_argument('-d', required=False, type=preproc_utils.str2bool, help='dot')

args = parser.parse_args()
first = args.f
last = args.l
if last == None:
    last = first+1
test = args.t
if test == None:
    test = False

dot = args.d
if dot == None:
    dot = True

exact_blobs = 0
over_blobs = 0
under_blobs = 0

exact_edges = 0
over_edges = 0
under_edges = 0

isomorphic_true = 0
isomorphic_false = 0

blob_number = ""
v_number = ""

edge_number = ""
real_edge_number = ""
time_table = []

blobs_total = 0
edges_total = 0
for j in range(2):
    file_pref = "graphs/planar/"
    if j == 0:
        print("PLANAR:")
    else:
        print("NON-PLANAR")
        file_pref = "graphs/nonplanar/"
    for i in range(first,last):
        file_string = file_pref + str(i) + "_*.jpg"
        for file in glob.glob(file_string):
            command = "python preproc.py -f " + file + " && python edge.py"
            start = time.time()
            output = subprocess.check_output(command, shell=True)
            end = time.time()
            time_table.append(end-start)
            for item in output.split("\n"):
                if "Blobs:" in item:
                    _, blob_number = item.split(" ")
                    v_number = re.split(r'[_.]', file)[3]
                    blobs_total += int(v_number)
                    if int(blob_number) < int(v_number):
                        under_blobs += 1
                    elif int(blob_number) == int(v_number):
                        exact_blobs += 1
                    else:
                        over_blobs += 1
                elif "Edges:" in item:
                    _, edge_number = item.split(" ")
                    real_edge_number = re.split(r'[_.]', file)[4]
                    edges_total += int(real_edge_number)
                    if int(edge_number) < int(real_edge_number):
                        under_edges += 1
                    elif int(edge_number) == int(real_edge_number):
                        exact_edges += 1
                    else:
                        over_edges += 1
            if test:
                print(output)
            else:
                number = (re.split(r'[_.]', file)[0]).split("/")[2]
                blob_status = "BLOBS OK"
                if int(blob_number) < int(v_number):
                    blob_status = "BLOBS UNDER"
                elif int(blob_number) > int(v_number):
                    blob_status = "BLOBS OVER"

                edge_status = "EDGES OK"
                if int(edge_number) < int(real_edge_number):
                    edge_status = "EDGES UNDER"
                elif int(edge_number) > int(real_edge_number):
                    edge_status = "EDGES OVER"

                isomorphic_status = "ISO OK"
                if dot:
                    G1 = NX.nx_agraph.read_dot("pics/dot")
                    G2 = NX.nx_agraph.read_dot("dots/" + number + ".dot")
                    GM = isomorphism.GraphMatcher(G1,G2)
                    result = GM.is_isomorphic()
                    isomorphic_status = "ISO OK"
                    if result:
                        isomorphic_true += 1
                    else:
                        isomorphic_false += 1
                        isomorphic_status = "ISO FALSE"

                print("Number " + number.ljust(2) +
                        " vertices: " + v_number.ljust(2) + " blobs: " + blob_number.ljust(2) + " " + blob_status.ljust(11) +
                        " edges: " + edge_number.ljust(2) + " real edges: " + real_edge_number.ljust(2) + " " + edge_status.ljust(11) +
                        " " + isomorphic_status.ljust(10) + " time: " + str(end-start))

                #command = "mkdir pics/" + number + " 2> /dev/null && cp pics/* pics/" + number + " 2> /dev/null"
                #try:
                #    output = subprocess.call(command, shell=True)
                #except:
                #    pass
    print("VERTICES: ", blobs_total, " EDGES: ", edges_total)

print("Exact blobs: " + str(exact_blobs))
print("Under blobs: " + str(under_blobs))
print("Over blobs: " + str(over_blobs))
print("Exact edges: " + str(exact_edges))
print("Under edges: " + str(under_edges))
print("Over edges: " + str(over_edges))
print("Isomorphic OK: " + str(isomorphic_true))
print("Isomorphic FALSE: " + str(isomorphic_false))
print("Total vertices: " + str(blobs_total))
print("Total edges: " +str(edges_total))

print("Threshold: " + str(config.threshold) + " Min sigma: " + str(config.min_sigma))
print("Avg time: " + str(np.average(time_table)) +
    " Min time: " + str(np.min(time_table)) +
    " Max time: " + str(np.max(time_table)))
