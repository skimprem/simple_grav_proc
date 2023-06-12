# Simple Gravimetric Processing

Simple processing of the observations with relative and absolute gravimeters

## Reqiurements

The following Python packages need to be installed:

- pandas
- numpy
- matplotlib
- seaborn

## Installing

### Windows system

1. Install [Python 3.x for Windows](https://www.python.org/downloads/windows/)
2. Install requirements packages

        pip install pandas numpy matplotlib seaborn

3. Clone this repository

        git clone https://github.com/skimprem/simple_grav_proc

4. From repository directory setup this package

        python setup.py install --user

### Linux system

        sudo python setyp.py install

## Usage

1. For Linux

        rgrav.py [options] path_to_input_filename.dat

        Options:
            -v                              verbose mode: print result table to stdout
            -o path_to_output_filename      custom name for report file (default: report_[input_filename].txt)
            --to_vgfit                      create CSV input file for the vg_fit utility
            --plot                          create residuals plot figure to PDF file (default show to display)

## To Do

- [ ] Moving to classes
- [ ] Loading set of the data files
- [ ] Option to select lines for processing
- [ ] Option to different processing methods
- [ ] Option to adjustment of network
