import argparse
from cg6_proc.utils import cg6_data_file_reader, get_readings, get_ties, get_mean_ties, get_ties_sum, print_means

parser = argparse.ArgumentParser(
                    prog='rgrav',
                    description='Read CG-6 data file and compute ties',
                    epilog='This program read CG-6 data file, then compute ties by ...')

parser.add_argument('data_file')                  # positional argument
# parser.add_argument('-c', '--count')                   # option that takes a value
parser.add_argument('-v', action='store_true')  # on/off flag

args = parser.parse_args()

data = cg6_data_file_reader(args.data_file) 
readings = get_readings(data)
ties = get_ties(readings)
means = get_mean_ties(ties)
sum = get_ties_sum(means)

if args.v:
    print_means(means)
    print(f'Sum of ties = {sum: .2f}')
else:
    print(sum)