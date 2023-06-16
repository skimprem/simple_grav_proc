'''
Read CG-6 data file and calculate ties
'''

import argparse
import os
from tkinter import filedialog as fd
import sys
from matplotlib import pyplot as plt
from cg6_proc.utils import \
    cg6_data_file_reader, \
    get_readings, \
    get_ties, \
    get_mean_ties, \
    make_vgfit_input, \
    get_report, \
    get_residuals_plot, \
    get_map

TO_VGFIT = False
VERBOSE = False
PLOT = False

# if sys.platform.startswith('linux'):
if sys.platform.startswith('win32'):
    data_file = fd.askopenfilename(
        defaultextension='.dat',
        filetypes=[('CG-6 data files', '*.dat'), ('All files', '*')],
        title='Choose data file'
    )
elif sys.platform.startswith('linux'):
    parser = argparse.ArgumentParser(
        prog='rgrav',
        description='Read CG-6 data file and compute ties',
        epilog='This program read CG-6 data file, then compute ties by ...'
    )
    parser.add_argument('data_file')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--to_vgfit', action='store_true')
    parser.add_argument('-o', metavar='out-file', type=argparse.FileType('w'))
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('-m', metavar='out-file', type=argparse.FileType('w'))
    args = parser.parse_args()
    data_file = args.data_file
    TO_VGFIT = args.to_vgfit
    VERBOSE = args.v
    PLOT = args.plot
    output_file_map = args.m
    output_file_report = args.o

data = cg6_data_file_reader(data_file)

readings = get_readings(data)
ties = get_ties(readings)
means = get_mean_ties(ties)

basename = os.path.splitext(os.path.basename(data_file))[0]

default_output_file_report = 'report_'+basename+'.txt'
if sys.platform.startswith('win32'):
    output_file_report = fd.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[('ACSII text file', '*.txt'), ('All files', '*')],
        initialfile=default_output_file_report,
        title="Save Report")
elif sys.platform.startswith('linux'):
    if not args.o:
        output_file_report = default_output_file_report
else:
    output_file_report = default_output_file_report

report = get_report(means)
with open(output_file_report, 'w', encoding='utf-8') as report_file:
    report_file.write(report)
    report_file.close()

if TO_VGFIT:
    make_vgfit_input(means, basename+'.csv')

if VERBOSE:
    print(report)

get_residuals_plot(data, readings, means)

if PLOT:
    plt.savefig(basename+'.pdf')
else:
    plt.show()


default_output_file_map = 'index_'+basename+'.html'
if sys.platform.startswith('win32'):
    output_file_map = fd.asksaveasfilename(
        defaultextension='.html',
        filetypes=[('html', '*.html'), ('All files', '*')],
        initialfile=default_output_file_map,
        title='Save Map')
elif sys.platform.startswith('linux'):
    if not args.m:
        output_file_map = default_output_file_map
else:
    output_file_man = default_output_file_map

get_map(readings).save(output_file_map)
