#!/usr/bin/env python
u'''
Read CG-6 data file and calculate vertical gradient
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from grav_proc.vertical_gradient import get_vg
from grav_proc.arguments import cli_vgrad_arguments, gui_vgrad_arguments
from grav_proc.loader import read_data
from grav_proc.calculations import make_frame_to_proc
from grav_proc.plots import vg_plot
from grav_proc.reports import make_vg_ties_report, make_vg_coeffs_report

def main():

    args = cli_vgrad_arguments()

    if args.input is None:
        args = gui_vgrad_arguments()
        if args.scale_factors:
            calibration_files = []
            for calibration_file_name in args.scale_factors:
                calibration_files.append(open(calibration_file_name, 'r', encoding='utf-8'))
            args.scale_factors = calibration_files

    data_files = []
    for data_file_name in args.input:
        data_files.append(open(data_file_name, 'r', encoding='utf-8'))
    args.input = data_files
    
    raw_data = make_frame_to_proc(read_data(args.input))

    vg_ties, vg_coef = get_vg(raw_data)

    if args.coeffs:
        output_coeffs = args.coeffs.name
    else:
        output_coeffs = 'coeffs.csv'

    make_vg_coeffs_report(vg_coef, output_coeffs, args.verbose)
    
    if args.ties:
        output_file = args.ties.name
    else:
        output_file = 'ties.csv'

    make_vg_ties_report(vg_ties, output_file, args.verbose)

    if args.plot:
        figs = vg_plot(vg_coef, vg_ties)
        for fig, filename in figs:
            fig.savefig(filename+'.png')
            
if __name__ == '__main__':
    main()