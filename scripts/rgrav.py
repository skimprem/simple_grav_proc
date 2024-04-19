#!/usr/bin/env python
u'''
Read CG-6 data file and calculate ties
'''

from tkinter import filedialog as fd
from matplotlib import pyplot as plt
import numpy as np
from grav_proc.arguments import cli_arguments, gui_arguments
from grav_proc.calculations import make_frame_to_proc, get_meters_readings, \
    get_meters_ties, get_meters_mean_ties, gravfit, to_minutes, get_vg, \
    get_meter_ties_by_lines, get_meter_ties_all #,vgfit2
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
    

    # fitgrav = gravfit(raw_data['station'], raw_data['corr_grav'], raw_data['std_err'], raw_data['date_time'].apply(to_minutes))

    # print(fitgrav)

    vg_proc = get_vg(raw_data)

    vg_ties, vg_coef = vg_proc

    for index, row in vg_coef.iterrows():
        df = vg_ties[(vg_ties.meter == row.meter) & (vg_ties.survey == row.survey)]
        h_f = df.from_height * 1e-3
        h_t = df.to_height * 1e-3
        y = np.linspace(0, 1.5, 50)
        p=np.poly1d(row.coefs[::-1]+[0])
        h = np.array(list(h_f)+list(h_t))
        h_min = min(h)
        h_ref = 1
        gp = lambda x: p(x) - x * (p(h_min) - p(h_ref)) / (h_min - h_ref)
        # gp = lambda x: p(x) - x * p(h_ref) / h_ref

        a, b = row.std_coefs
        u = abs(y - h_ref) * np.sqrt(a**2 + (y - h_ref)**2 * b**2 + 2 * (y - h_ref) * row.cov_coefs)
        x = gp(y)
        plt.plot(x, y)
        plt.fill_betweenx(y, x - u, x + u, alpha=0.2)

        vg = df.gravity / (h_t - h_f)
        mean_vg = np.mean(vg)
        
        for f, t, vg in zip(h_f, h_t, vg):
            x1, x2 = f * vg - p(h_ref) * f / h_ref, t * vg - p(h_ref) * t / h_ref
            # x1, x2 = f * vg - mean_vg * f, t * vg - mean_vg * t
            x11, x22 = p(f) - p(h_ref) * f, p(t) - p(h_ref) * t
            print('meas', f, x1, t, x2)
            # print('model', x11, x22)
            # print('diff', x11-x1, x22-x2)
            plt.plot([x11, x22], [f, t], 'o', [x1, x2], [f, t], '.-')
 
        plt.title(f'Vertical gradient model for meter {row.meter} (substract {p(h_ref):.2f} $\\times$ height)')
        plt.xlabel(f'Gravity, $\mu$Gal')
        plt.ylabel('Height, m')

        plt.show()
    
    readings = get_meters_readings(raw_data)

    ties = get_meters_ties(readings)

    # print(get_meter_ties_by_lines(raw_data))
    # print(get_meter_ties_all(raw_data))
    means = get_meters_mean_ties(ties)

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

    report = get_report(means)

    output_file_report.write(report)
    output_file_report.close()

    if not gui_mode:
        if args.to_vgfit:
            make_vgfit_input(means, 'output.vg')
        if args.verbose:
            print(report)

    if gui_mode:
        get_residuals_plot(raw_data, readings, means)
        plt.show()
    else:
        if args.plot:
            get_residuals_plot(raw_data, readings, means)
            plt.savefig(basename+'.pdf')

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

    #     get_map(readings).save(output_file_map)

# run main program
if __name__ == '__main__':
    main()