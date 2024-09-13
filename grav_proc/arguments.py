import argparse

from tkinter import filedialog as fd
from tkinter import simpledialog as sd
from tkinter import messagebox as mb

def cli_rgrav_arguments():

    parser = argparse.ArgumentParser(
        prog='rgrav',
        description='Read CG-6 data file and compute ties',
        epilog='This program read CG-6 data file, then compute ties by ...',
        exit_on_error=False
    )

    parser.add_argument(
        '--method',
        type=str,
        help='LS method: WLS (default) or RLM'
    )

    parser.add_argument(
        '--by_lines',
        action='store_true',
        help='Calc by lines'
    )

    parser.add_argument(
        '--input',
        nargs='+',
        help='Input data files'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print results to stdout'
    )

    parser.add_argument(
        '--to_vgfit',
        action='store_true',
        help='Create CSV file for the vg_fit utility'
    )

    parser.add_argument(
        '--output',
        metavar='out-file',
        type=argparse.FileType('w'),
        help='Name for the report file'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create plot to PDF'
    )

    parser.add_argument(
        '--scale_factors',
        type=argparse.FileType('r'),
        nargs='+',
        help='Calibration factors for all gravimeters'
    )

    parser.add_argument(
        '--map',
        action='store_true',
        help='Name for the map file'
    )

    parser.add_argument(
        '--anchor',
        type=str,
        help='Fix Station'
    )

    return parser.parse_args()

def gui_rgrav_arguments():
    
    method=sd.askstring(
        title='Solve method',
        initialvalue='WLS',
        prompt='Enter method (WLS or RLM):',
    )
    
    data_file_names = fd.askopenfilenames(
        defaultextension='.dat',
        filetypes=[('CG-6 data files', '*.dat'), ('All files', '*')],
        title='Choose data file'
    )
    
    arguments = []
    parser = argparse.ArgumentParser()

    scale_factors_mode = mb.askyesno(
        title='Calibration file selected',
        message='Want to load a calibration factors?'
    )

    if scale_factors_mode:
        scale_factors = fd.askopenfilenames(
            defaultextension='.txt',
            filetypes=[('Calibration files', '*.txt'), ('All files', '*')],
            title='Choose data file'
        )

    by_lines_mode = mb.askyesno(
        title='Calc by lines',
        message='Want to calc by lines?'
    )

    parser.add_argument('--method', type=str)
    arguments.append('--method')
    arguments.append(method)

    parser.add_argument('--input')
    arguments.append('--input')
    arguments.append(data_file_names)

    parser.add_argument('--scale_factors')
    if scale_factors_mode:
        arguments.append('--scale_factors')
        arguments.append(scale_factors)

    parser.add_argument('--by_lines', action='store_true')
    if by_lines_mode:
        arguments.append('--by_lines')

    parser.add_argument('--anchor', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--map', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args(arguments)
    

def cli_vgrad_arguments():

    parser = argparse.ArgumentParser(
        prog='vgrad',
        description='Read CG-6 data file and compute vertical gradient',
        epilog='This program read CG-6 data file, then compute vertical gradient by using least square methods',
        exit_on_error=False
    )

    parser.add_argument(
        '--input',
        nargs='+',
        help='Input data files'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print results to stdout'
    )

    parser.add_argument(
        '--coeffs',
        metavar='out-file',
        type=argparse.FileType('w'),
        help='Name for the coeffs file'
    )

    parser.add_argument(
        '--ties',
        metavar='out-file',
        type=argparse.FileType('w'),
        help='Name for the ties file'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create plot to PNG'
    )

    parser.add_argument(
        '--scale_factors',
        type=argparse.FileType('r'),
        nargs='+',
        help='Calibration factors for all gravimeters'
    )

    return parser.parse_args()


def gui_vgrad_arguments():
    
    data_file_names = fd.askopenfilenames(
        defaultextension='.dat',
        filetypes=[('CG-6 data files', '*.dat'), ('All files', '*')],
        title='Choose data file'
    )
    
    arguments = []
    parser = argparse.ArgumentParser()

    scale_factors_mode = mb.askyesno(
        title='Calibration file selected',
        message='Want to load a calibration factors?'
    )
    
    if scale_factors_mode:
        scale_factors = fd.askopenfilenames(
            defaultextension='.txt',
            filetypes=[('Calibration files', '*.txt'), ('All files', '*')],
            title='Choose data file'
        )
    
    coeffs = fd.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[('Vertical gradient coefficients', '*.csv'), ('All files', '*')],
        title='Save coefficients file'
    )

    ties = fd.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[('Vertical gradient ties', '*.csv'), ('All files', '*')],
        title='Save ties file'
    )


    plot_mode = mb.askyesno(
        title='Plot results',
        message='Want to plot a vertical gradient?'
    )
    
    verbose_mode = mb.askyesno(
        title='Verbose mode',
        message='Want to verbose mode?'
    )
 
    parser.add_argument('--input')
    arguments.append('--input')
    arguments.append(data_file_names)

    parser.add_argument('--coeffs', type=argparse.FileType('w'))
    arguments.append('--coeffs')
    arguments.append(coeffs)

    parser.add_argument('--ties', type=argparse.FileType('w'))
    arguments.append('--ties')
    arguments.append(ties)

    parser.add_argument('--plot', action='store_true')
    if plot_mode:
        arguments.append('--plot')

    parser.add_argument('--verbose', action='store_true')
    if plot_mode:
        arguments.append('--verbose')

    parser.add_argument('--scale_factors')
    if scale_factors_mode:
        arguments.append('--scale_factors')
        arguments.append(scale_factors)

    return parser.parse_args(arguments)
 