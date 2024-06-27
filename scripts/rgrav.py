#!/usr/bin/env python
u'''
Read CG-6 data file and calculate ties
'''

from tkinter import filedialog as fd
from grav_proc.arguments import cli_rgrav_arguments, gui_rgrav_arguments
from grav_proc.calculations import make_frame_to_proc, \
    fit_by_meter_created
from grav_proc.loader import read_data, read_scale_factors
from grav_proc.plots import residuals_plot, get_map
from grav_proc.reports import get_report #, make_vgfit_input

def main():

    gui_mode = False
    args = cli_rgrav_arguments()

    if args.input is None:
        gui_mode = True
        args = gui_rgrav_arguments()
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

    if args.scale_factors:
        scale_factors = read_scale_factors(args.scale_factors)
        group_by_meter = scale_factors.groupby('instrument_serial_number')
        for meter, meter_scale_factors in group_by_meter:
            scale_factors = list(meter_scale_factors['scale_factor'])
            scale_factors_std = list(meter_scale_factors['scale_factor'])
            if len(scale_factors) > 1:
                print('Warning: There is more than one scale factor for a current gravity meter!')
            scale_factor = scale_factors[0]
            scale_factor_std = scale_factors_std[0]
            raw_data.loc[raw_data['instrument_serial_number'] == meter, 'scale_factor'] = scale_factor
            raw_data.loc[raw_data['instrument_serial_number'] == meter, 'scale_factor_std'] = scale_factor_std
            raw_data.loc[raw_data['instrument_serial_number'] == meter, 'corr_grav'] = raw_data.loc[raw_data['instrument_serial_number'] == meter, 'corr_grav'] * scale_factor
    
    by_lines = False
    if args.by_lines:
        by_lines = True

    method = 'WLS'
    if args.method:
        method = args.method

    anchor = None
    if args.anchor:
        anchor = args.anchor

    ties = fit_by_meter_created(raw_data, anchor=anchor, method=method, by_lines=by_lines)

    basename = '-'.join(str(survey) for survey in raw_data.station.unique())

    if args.plot:
        fig = residuals_plot(raw_data)
        fig.savefig(f'{basename}.png')
        fig.show()

    default_output_file_report = 'report_'+basename+'.txt'

    if gui_mode:
        output_file_report = open(fd.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[('ACSII text file', '*.txt'), ('All files', '*')],
            initialfile=default_output_file_report,
            title="Save Report"), 'w', encoding='utf-8')
    else:
        if args.output:
            output_file_report = args.output
        else:
            output_file_report = open(default_output_file_report, 'w', encoding='utf-8')

    report = get_report(ties)

    output_file_report.write(report)
    output_file_report.close()

    if args.verbose:
        print(report)
        
    if args.map:
        fig = get_map(ties)
        fig.savefig(f'{basename}.pdf', bbox_inches = 'tight')

# run main program
if __name__ == '__main__':
    main()