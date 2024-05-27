#!/usr/bin/env python
u'''
Read CG-6 data file and calculate ties
'''

from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import pandas as pd
from grav_proc.arguments import cli_rgrav_arguments, gui_rgrav_arguments
from grav_proc.calculations import make_frame_to_proc, gravfit, to_days, drift_fitting, \
    get_meters_readings, get_meters_ties, get_meters_mean_ties
from grav_proc.loader import read_data, read_scale_factors
from grav_proc.plots import get_residuals_plot, get_map
from grav_proc.reports import get_report, make_vgfit_input

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
    # raw_data['station'].astype(str)

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
    
    ties = pd.DataFrame()
    meters = raw_data['instrument_serial_number'].unique()
    meter_number = {}
    for index, meter in enumerate(meters):
        meter_number[meter] = index
    fig, ax = plt.subplots(nrows=len(meters), figsize=(16, 8), layout='constrained')
    fig.supylabel('Residuals, $\mu$Gal')
    fig.supxlabel('Date Time')
    for meter_created, grouped in raw_data.groupby(['instrument_serial_number', 'created']):
        meter, created = meter_created
        fitgrav, grouped['resid'] = gravfit(grouped['station'], grouped['corr_grav'], grouped['std_err'], grouped['date_time'].apply(to_days), args.anchor)
        fitgrav['meter'] = meter
        fitgrav['survey'] = created.date()
        ties = pd.concat([ties, fitgrav], ignore_index=True)
        for station, grouped_by_station in grouped.groupby('station'):
            if len(meters) > 1:
                ax[meter_number[meter]].set_title(f'CG-6 #{meter}', loc='left')
                ax[meter_number[meter]].plot(grouped_by_station['date_time'], grouped_by_station['resid'], '.', label=station)
                ax[meter_number[meter]].legend(loc='upper right')
            else:
                ax.set_title(f'CG-6 #{meter}', loc='left')
                ax.plot(grouped_by_station['date_time'], grouped_by_station['resid'], '.', label=station)
                ax.legend(loc='upper right')
    print(ties)
    # fig.tight_layout()
    fig.savefig('output.png')
    plt.show()
   
    # readings = get_meters_readings(raw_data)

    # ties = get_meters_ties(readings)
    # print(ties)

    # print(get_meter_ties_by_lines(raw_data))
    # print(get_meter_ties_all(raw_data))
    # means = get_meters_mean_ties(ties)
    # print(means)

    basename = '-'.join(str(survey) for survey in raw_data.station.unique())

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

    # report = get_report(fitgrav)

    # output_file_report.write(report)
    # output_file_report.close()

    # if not gui_mode:
        # if args.to_vgfit:
            # make_vgfit_input(vg_ties, 'output.vg')
        # if args.verbose:
            # print(report)

    # if gui_mode:
    #     get_residuals_plot(raw_data, readings, means)
    #     plt.show()
    # else:
    #     if args.plot:
    #         get_residuals_plot(raw_data, readings, means)
    #         plt.savefig(basename+'.pdf')

    # default_output_file_map = 'index_'+basename+'.html'

    # if not args.to_vgfit:
    #     if gui_mode:
    #         output_file_map = fd.asksaveasfilename(
    #             defaultextension='.html',
    #             filetypes=[('html', '*.html'), ('All files', '*')],
    #             initialfile=default_output_file_map,
    #             title='Save Map')
    #     else:
    #         if args.map:
    #             output_file_map = args.map
    #         else:
    #             output_file_map = default_output_file_map

        # get_map(readings).save(output_file_map)

# run main program
if __name__ == '__main__':
    main()