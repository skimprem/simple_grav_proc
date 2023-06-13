import argparse
import os
from cg6_proc.utils import cg6_data_file_reader, get_readings, get_ties, get_mean_ties, \
make_vgfit_input, get_report, sort_ties, get_residuals_plot, get_map
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import sys

to_vgfit = False
v = False
s = False
p = False
m = False

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
                    epilog='This program read CG-6 data file, then compute ties by ...')

    parser.add_argument('data_file')                        # positional argument
    parser.add_argument('-v', action='store_true')          # on/off flag
    parser.add_argument('--to_vgfit', action='store_true')  # on/off flag
    parser.add_argument('-s', action='store_true')          # on/off flag
    parser.add_argument('-o', metavar='out-file', type=argparse.FileType('w'))
    parser.add_argument('--plot', action='store_true')      # on/off flag
    parser.add_argument('--map', action='store_true')      # on/off flag
    args = parser.parse_args()
    data_file = args.data_file
    to_vgfit = args.to_vgfit 
    v = args.v 
    s = args.s
    p = args.plot
    m = args.map
    output_file = args.o 

data = cg6_data_file_reader(data_file) 

readings = get_readings(data)
ties = get_ties(readings)
means = get_mean_ties(ties)
if s:
    means = sort_ties(means)

basename = os.path.splitext(os.path.basename(data_file))[0]

default_output_file = 'report_'+basename+'.txt'
# if sys.platform.startswith('linux'):
if sys.platform.startswith('win32'):
    output_file = fd.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[('ACSII text file', '*.txt'), ('All files', '*')],
        initialfile=default_output_file,
        title="Save Report")
elif sys.platform.startswith('linux'):
    if not args.o:
        output_file = default_output_file
else:
    output_file = default_output_file
    
report = get_report(means)
with open(output_file, 'w') as report_file:
    report_file.write(report)
    report_file.close()

if to_vgfit:
    make_vgfit_input(means, basename+'.csv')

if v:
    print(report)

get_residuals_plot(data, readings, means)

if p:
    plt.savefig(basename+'.pdf')
else:
    plt.show()

if m:
    map = get_map(readings)
    map.save(basename+'.html')