'''
Set of utilites for relative gravity processing
'''

from datetime import datetime as dt
import re
import pandas as pd

def format_detect(data_file):
    for line in data_file:
        line = line.strip()
        if not line or line == '/':
            continue
        if line[0] == '/':
            line = line[1:].strip()
            match line.split()[0]:
                case 'CG-6':
                    data_file.seek(0)
                    return 'cg6'
                case 'CG-5':
                    data_file.seek(0)
                    return 'cg5'
                case _:
                    data_file.seek(0)
                    return None


def read_data(data_files):
    ''' Load data from CG-6 data file'''

    match format_detect(data_files[0]):
        case 'cg5':
            cg5_reader(data_files)
        case 'cg6':
            return cg6_reader(data_files)
        case _:
            raise ImportError(f'{data_files[0].name} data file must be in CG-x format')


def cg5_reader(data_files):

    meter_type = 'cg5'

    rows = {
        'Survey name': [],
        'Instrument S/N': [],
        'Client': [],
        'Operator': [],
        'Date': [],
        'Time': [],
        'LONG': [],
        'LAT': [],
        'ZONE': [],
        'GMT DIFF.': [],
        'Gref': [],
        'Gcal1': [],
        'TiltxS': [],
        'TiltyS': [],
        'TiltxO': [],
        'TiltyO': [],
        'Tempco': [],
        'Drift': [],
        'DriftTime Start': [],
        'DriftDate Start': [],
        'Tide Correction': [],
        'Cont. Tilt': [],
        'Auto Rejection': [],
        'Terrain Corr.': [],
        'Seismic Filter': [],
        'Raw Data': [],
        'LINE': [],
        'STATION': [],
        'ALT.': [],
        'GRAV.': [],
        'SD.': [],
        'TILTX': [],
        'TILTY': [],
        'TEMP': [],
        'TIDE': [],
        'DUR': [],
        'REJ': [],
        'TIME': [],
        'DEC.TIME+DATE': [],
        'TERRAIN': [],
        'DATE': [],
        'MeterType': [],
        'DataFile': [],
    }

    for data_file in data_files:

        if format_detect(data_file) != meter_type:
            raise ImportError(f'{data_file.name} data file must be in {meter_type.upper()} format')
        
        lines = data_file.readlines()
        data_file.close()

        line_station_format = False

        row = {}

        for line in lines:
            if not line.strip():
                continue
            match line[:2].strip():
                case '/':
                    split_line = re.split(':', line[1:])
                    for index in range(len(split_line)):
                        split_line[index] = split_line[index].strip()
                    if len(split_line) > 1:
                        row.update({split_line[0]: ' '.join(x for x in split_line[1:])})
                case '/-':
                    line = line[1:].replace('-', ' ')
                    headers = line.split()
                case 'Li':
                    line_station_format = True
                case _:
                    split_line = line.split()
                    for index in range(len(split_line)):
                        row.update({headers[index]: split_line[index]})
                    row.update({'MeterType': meter_type.upper()})
                    row.update({'DataFile': data_file.name})
                    
                    for key, value in row.items():
                        rows[key].append(value)

    cg_data = pd.DataFrame(rows)

    exit()



def cg6_reader(data_files):
    meter_type = 'cg6'
    rows = {
        'Survey Name': [],
        'Instrument Serial Number': [],
        'Created': [],
        'Operator': [],
        'Gcal1 [mGal]': [],
        'Goff [ADU]': [],
        'Gref [mGal]': [],
        'X Scale [arc-sec/ADU]': [],
        'Y Scale [arc-sec/ADU]': [],
        'X Offset [ADU]': [],
        'Y Offset [ADU]': [],
        'Temperature Coefficient [mGal/mK]': [],
        'Temperature Scale [mK/ADU]': [],
        'Drift Rate [mGal/day]': [],
        'Drift Zero Time': [],
        'Firmware Version': [],
        'Station': [],
        'Date': [],
        'Time': [],
        'CorrGrav': [],
        'Line': [],
        'StdDev': [],
        'StdErr': [],
        'RawGrav': [],
        'X': [],
        'Y': [],
        'SensorTemp': [],
        'TideCorr': [],
        'TiltCorr': [],
        'TempCorr': [],
        'DriftCorr': [],
        'MeasurDur': [],
        'InstrHeight': [],
        'LatUser': [],
        'LonUser': [],
        'ElevUser': [],
        'LatGPS': [],
        'LonGPS': [],
        'ElevGPS': [],
        'Corrections[drift-temp-na-tide-tilt]': [],
        'MeterType': [],
        'DataFile': []
    }

    for data_file in data_files:
        if format_detect(data_file) != meter_type:
            raise ImportError(f'{data_file.name} data file must be in {meter_type.upper()} format')

        lines = data_file.readlines()
        data_file.close()
            
        row = {}
        for line in lines:
            if not line.strip():
                continue
            match line[:2].strip():
                case '/':
                    split_line = re.split(':', line[1:])
                    for index in range(len(split_line)):
                        split_line[index] = split_line[index].strip()
                    if len(split_line) > 1:
                        row.update({split_line[0]: ' '.join(x for x in split_line[1:])})
                case '/S':
                    line = line[1:]
                    headers = line.split()
                case _:
                    split_line = line.split()
                    for index in range(len(split_line)):
                        row.update({headers[index]: split_line[index]})
                    row.update({'MeterType': meter_type.upper()})
                    row.update({'DataFile': data_file.name})
                    
                    for key, value in row.items():
                        rows[key].append(value)    
                
    cg_data = pd.DataFrame(rows)        

    cg_data = cg_data.astype({
        'Instrument Serial Number': 'int',
        'Gcal1 [mGal]': 'float',
        'Goff [ADU]': 'float',
        'Gref [mGal]': 'float',
        'X Scale [arc-sec/ADU]': 'float',
        'Y Scale [arc-sec/ADU]': 'float',
        'X Offset [ADU]': 'float',
        'Y Offset [ADU]': 'float',
        'Temperature Coefficient [mGal/mK]': 'float',
        'Temperature Scale [mK/ADU]': 'float',
        'Drift Rate [mGal/day]': 'float',
        'CorrGrav': 'float',
        'Line': 'int',
        'StdDev': 'float',
        'StdErr': 'float',
        'RawGrav': 'float',
        'X': 'float',
        'Y': 'float',
        'SensorTemp': 'float',
        'TideCorr': 'float',
        'TiltCorr': 'float',
        'TempCorr': 'float',
        'DriftCorr': 'float',
        'MeasurDur': 'int',
        'InstrHeight': 'float',
        'LatUser': 'float',
        'LonUser': 'float',
        'ElevUser': 'float',
        'LatGPS': 'float',
        'LonGPS': 'float',
        'ElevGPS': 'float',
        }, errors='ignore')
    cg_data['Created'] = pd.to_datetime(cg_data['Created'], format='%Y-%m-%d %H %M %S')
    cg_data['Drift Zero Time'] = pd.to_datetime(cg_data['Drift Zero Time'], format='%Y-%m-%d %H %M %S')
    for index, row in cg_data.iterrows():
        cg_data.loc[index, 'date_time'] = dt.strptime(row.Date+' '+row.Time, '%Y-%m-%d %H:%M:%S')

    # cg_data = cg_data.set_index('date_time')
    return cg_data

def read_calibration_factors(calibration_files):
    
    ''' Get calibration factors from file(s) '''
    
    calibration_factors = pd.DataFrame()

    for calibration_file in calibration_files:
        calibration_data = pd.read_csv(calibration_file.name, delimiter=' ', names = ['meter', 'k', 'std_k'])
        calibration_factors = pd.concat([calibration_factors, calibration_data], axis=0)

    calibration_factors = calibration_factors.reset_index()

    return calibration_factors
    