#!/usr/bin/python

from CommonUtil.FunctionsUtil import *
from collections import *
import numpy as np
import sys


if( len(sys.argv)!=2 ):
    print("ERROR. Please input the input file name... EXIT")
    sys.exit(0)
else:
    in_filename = str(sys.argv[1])

print("Compute mean of data from file: %s..." %(basename(in_filename)))

out_fullfilename = joinpathnames(dirnamepathfile(in_filename), 'mean_' + basename(in_filename))


input_cases_names = ['av24', 'av25', 'av26', 'av28', 'av41']
num_input_cases = len(input_cases_names)


raw_data_string = np.genfromtxt(in_filename, dtype=str)
raw_data_float  = np.genfromtxt(in_filename, dtype=float)

fields_names = [item.replace('/','')  for item in raw_data_string[0,1:]]
cases_names  = [item.replace('\'','') for item in raw_data_string[1:,0]]
data         = raw_data_float[1:,1:]

num_fields = len(fields_names)

indexes_input_cases = []

for in_case in input_cases_names:
    if in_case in cases_names:
        indexes_input_cases.append(cases_names.index(in_case))
    else:
        print("ERROR: case '%s' not found... EXIT" %(in_case))
        sys.exit(0)

print("Compute mean of data for fields: %s..." %(', '.join(map(lambda item: '/'+item+'/', fields_names))))
print("Compute mean of data for cases: %s..." %(', '.join(map(lambda item: '/'+item+'/', input_cases_names))))


# allocate vars to store data in files and compute the mean
mean_data_fields_files = np.mean(data[indexes_input_cases,:], axis=0)


print("Save in file: '%s'..." %(out_fullfilename))

fout = open(out_fullfilename, 'w')

strheader = '/case/ ' + ' '.join(['/%s/'%(elem) for elem in fields_names]) +'\n'
fout.write(strheader)

strdata = '\'mean\'' + ' ' + ' '.join([str(elem) for elem in mean_data_fields_files]) + '\n'
fout.write(strdata)

fout.close()