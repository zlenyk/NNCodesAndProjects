import sys
import argparse
import glob
import subprocess
import preproc_utils

parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, type=str, help='first file')
parser.add_argument('-t', required=False, type=preproc_utils.str2bool, help='test')


args = parser.parse_args()

test = args.t
if test == None:
    test = False

file = args.f

command = "python preproc.py -f " + file + " -t " +str(test) + " && python edge.py -t" + str(test)
output = subprocess.check_output(command, shell=True)
print(output)
command = "cp pics/dot pics/dot.pdf results/"
output = subprocess.check_output(command, shell=True)

if test:
    command = "open pics/dot.pdf"
    output = subprocess.check_output(command, shell=True)
