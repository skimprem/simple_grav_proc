import argparse

from tkinter import filedialog as fd
from tkinter import simpledialog as sd
from tkinter import messagebox as mb

def cli_arguments():

    parser = argparse.ArgumentParser(
        prog='rgrav',
        description='Read CG-6 data file and compute ties',
        epilog='This program read CG-6 data file, then compute ties by ...',
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
        metavar='out-file',
        type=argparse.FileType('w'),
        help='Name for the map file'
    )

    return parser.parse_args()

def gui_arguments():
    
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

    parser.add_argument('--input')
    arguments.append('--input')
    arguments.append(data_file_names)

    parser.add_argument('--scale_factors')
    if scale_factors_mode:
        arguments.append('--scale_factors')
        arguments.append(scale_factors)

    return parser.parse_args(arguments)
