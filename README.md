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

        usage: rgrav [-h] [--verbose] [--to_vgfit] [--output out-file] [--plot] [--map out-file] data_files [data_files ...]

        Read CG-6 data file and compute ties

        positional arguments:
          data_files

        options:
          -h, --help         show this help message and exit
          --verbose          Print results to stdout
          --to_vgfit         Create CSV file for the vg_fit utility
          --output out-file  Name for the report file
          --plot             Create plot to PDF
          --map out-file     Name for the map file

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
