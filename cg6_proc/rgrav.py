import argparse
import os
from cg6_proc.utils import cg6_data_file_reader, get_readings, get_ties, get_mean_ties, get_ties_sum, print_means, make_vgfit_input, make_output

parser = argparse.ArgumentParser(
                    prog='rgrav',
                    description='Read CG-6 data file and compute ties',
                    epilog='This program read CG-6 data file, then compute ties by ...')

parser.add_argument('data_file')                  # positional argument
parser.add_argument('-v', action='store_true')  # on/off flag
parser.add_argument('--to_vgfit', action='store_true')  # on/off flag

args = parser.parse_args()

data = cg6_data_file_reader(args.data_file) 
readings = get_readings(data)
ties = get_ties(readings)
means = get_mean_ties(ties)
sum = get_ties_sum(means)

basename = os.path.splitext(os.path.basename(args.data_file))[0]
dirname = os.path.dirname(args.data_file)

output_file = 'report_'+basename+'.txt'
make_output(means, output_file)
report = open(output_file, 'a')
report.write(f'\n Sum of the ties = {get_ties_sum(means): .2f}\n')
report.close()

if args.to_vgfit:
    make_vgfit_input(means, basename+'.csv')

if args.v:
    print_means(means)
    print(f'Sum of the ties = {sum: .2f}')

input()