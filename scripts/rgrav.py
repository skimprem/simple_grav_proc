#!/usr/bin/env python
u'''
Read CG-6 data file and calculate ties
'''

from tkinter import filedialog as fd
from matplotlib import pyplot as plt
from grav_proc.arguments import cli_arguments, gui_arguments
from grav_proc.calculations import make_frame_to_proc, get_meters_readings, \
    get_meters_ties, get_meters_mean_ties
from grav_proc.loader import read_data, read_scale_factors
from grav_proc.plots import get_residuals_plot, get_map
from grav_proc.reports import get_report, make_vgfit_input

def main():

    gui_mode = False
    args = cli_arguments()

    if args.input is None:
        gui_mode = True
        args = gui_arguments()
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
    
    readings = get_meters_readings(raw_data)
    ties = get_meters_ties(readings)
    means = get_meters_mean_ties(ties)

    basename = '_'.join(str(survey) for survey in raw_data.survey_name.unique())

    default_output_file_report = 'report_'+basename+'.txt'

    if gui_mode:
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

    report = get_report(means)

    print(output_file_report)

    with open(output_file_report, 'w', encoding='utf-8') as report_file:
        report_file.write(report)
        report_file.close()

    if not gui_mode:
        if args.to_vgfit:
            make_vgfit_input(means)
        if args.verbose:
            print(report)


    if gui_mode:
        get_residuals_plot(raw_data, readings, means)
        plt.show()
    else:
        if args.plot:
            get_residuals_plot(raw_data, readings, means)
            plt.savefig(basename+'.pdf')

    default_output_file_map = 'index_'+basename+'.html'

    if not args.to_vgfit:
        if gui_mode:
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

# run main program
if __name__ == '__main__':
    main()