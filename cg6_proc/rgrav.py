import argparse
import os
from cg6_proc.utils import cg6_data_file_reader, get_readings, get_ties, get_mean_ties, \
get_ties_sum, print_means, make_vgfit_input, make_output, sort_ties
from tkinter import filedialog as fd
import sys

to_vgfit = False
v = False
s = False

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

    parser.add_argument('data_file')                  # positional argument
    parser.add_argument('-v', action='store_true')  # on/off flag
    parser.add_argument('--to_vgfit', action='store_true')  # on/off flag
    parser.add_argument('-s', action='store_true')  # on/off flag
    parser.add_argument('-o', metavar='out-file') #, type=argparse.FileType('w'))
    args = parser.parse_args()
    data_file = args.data_file
    to_vgfit = args.to_vgfit 
    v = args.v 
    s = args.s
    output_file = args.o 

data = cg6_data_file_reader(data_file) 

readings = get_readings(data)
ties = get_ties(readings)
means = get_mean_ties(ties)
if s:
    means = sort_ties(means)
sum = get_ties_sum(means)

basename = os.path.splitext(os.path.basename(data_file))[0]
# dirname = os.path.dirname(data_file)

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
    
make_output(means, output_file)
with open(output_file, 'a') as report:
    report.write(f'\n Sum of the ties = {get_ties_sum(means): .2f}\n')
    report.close()

if to_vgfit:
    make_vgfit_input(means, basename+'.csv')

if v:
    print_means(means)
    print(f'Sum of the ties = {sum: .2f}')