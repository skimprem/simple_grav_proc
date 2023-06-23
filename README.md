# Simple Gravimetric Processing

Simple processing of the observations with relative and absolute gravimeters

## Reqiurements

The following Python packages need to be installed:

- pandas
- numpy
- matplotlib
- seaborn
- geopandas
- folium
- mapclassify

## Installing

### Windows system

1. Install [Python 3.x for Windows](https://www.python.org/downloads/windows/)
<!-- 2. Install requirements packages

        pip install pandas numpy matplotlib seaborn -->

2. Clone this repository

        git clone https://github.com/skimprem/simple_grav_proc

3. Install this package from the repository directory

        pip install .

### Linux system

        sudo python setup.py install

## Usage

1. For Linux

        rgrav.py [options] <path to input filename>.dat

        Options:
            -v                              verbose mode: print result table to stdout
            -o path_to_output_report        custom name for report file (default: report_[input_filename].txt)
            -m path_to_output_map           custom name for map file (default: index_[input_filename].txt)
            --to_vgfit                      create CSV input file for the vg_fit utility
            --plot                          create residuals plot figure to PDF file (default show to display)

## To Do

- [ ] Moving to classes
- [x] Loading set of the data files
- [ ] Option to select lines for processing
- [ ] Option to different processing methods
- [ ] Option to adjustment of network
- [ ] Test of Windows installing
- [x] Option to interactive stations mapping (GeoPandas, Leaflet, Folium etc.)
- [ ] Correct setting for setuptools
- [ ] Loading CG-5 data files
- [x] Processing with set of meters
