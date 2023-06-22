'''
Read CG-6 data file and calculate ties
'''

import argparse
from tkinter import filedialog as fd
import sys
from matplotlib import pyplot as plt
from cg6_proc.utils import \
    read_data, \
    get_readings, \
    get_ties, \
    get_mean_ties, \
    make_vgfit_input, \
    get_report, \
    get_residuals_plot, \
    get_map, \
    make_frame_to_proc, \
    get_meters_readings, \
    get_meters_ties, \
    get_meters_mean_ties, \
    get_meters_report

GUI = False

if sys.platform.startswith('win32'):
    GUI = True

if GUI:
    data_file_names = fd.askopenfilenames(
        defaultextension='.dat',
        filetypes=[('CG-6 data files', '*.dat'), ('All files', '*')],
        title='Choose data file'
    )
    data_files = []
    for data_file_name in list(data_file_names):
        data_files.append(open(data_file_name, 'r'))
else:
    parser = argparse.ArgumentParser(
        prog='rgrav',
        description='Read CG-6 data file and compute ties',
        epilog='This program read CG-6 data file, then compute ties by ...',
        exit_on_error=False
    )
    parser.add_argument('data_files', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('--verbose', action='store_true', help='Print results to stdout')
    parser.add_argument('--to_vgfit', action='store_true', help='Create CSV file for the vg_fit utility')
    parser.add_argument('--output', metavar='out-file', type=argparse.FileType('w'), help='Name for the report file')
    parser.add_argument('--plot', action='store_true', help='Create plot to PDF')
    # parser.add_argument('--gui', action='store_true', help='GUI mode')
    parser.add_argument('--map', metavar='out-file', type=argparse.FileType('w'), help='Name for the map file')

    args = parser.parse_args()

    data_files = args.data_files
    # GUI=args.gui

raw_data = make_frame_to_proc(read_data(data_files))

readings = get_meters_readings(raw_data)
ties = get_meters_ties(readings)
means = get_meters_mean_ties(ties)

basename = '_'.join(str(survey) for survey in raw_data.survey_name.unique())

default_output_file_report = 'report_'+basename+'.txt'

if GUI:
    output_file_report = fd.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[('ACSII text file', '*.txt'), ('All files', '*')],
        initialfile=default_output_file_report,
        title="Save Report")
else:
    if args.output:
        output_file_report = args.output
    else:
        output_file_report = default_output_file_report

report = get_meters_report(means)

with open(output_file_report, 'w', encoding='utf-8') as report_file:
    report_file.write(report)
    report_file.close()

if not GUI:
    if args.to_vgfit:
        for meter in means.instrument_serial_number.unique():
            meter_means = means[means.instrument_serial_number == meter]
            make_vgfit_input(means, str(meter)+'.csv')

    if args.verbose:
        print(report)

for meter in means.instrument_serial_number.unique():
    meter_means = means[means.instrument_serial_number == meter]
    meter_readings = readings[readings.instrument_serial_number == meter]
    meter_data = raw_data[raw_data.instrument_serial_number == meter]
    get_residuals_plot(meter_data, meter_readings, meter_means)

default_output_file_map = 'index_'+basename+'.html'

if GUI:
    output_file_map = fd.asksaveasfilename(
        defaultextension='.html',
        filetypes=[('html', '*.html'), ('All files', '*')],
        initialfile=default_output_file_map,
        title='Save Map')
else:
    if args.map:
        output_file_map = args.map
    else:
        output_file_map = default_output_file_map

get_map(readings).save(output_file_map)
