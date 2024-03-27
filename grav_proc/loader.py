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
            return cg5_to_cg6_converter(cg5_reader(data_files))
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

    cg_data = cg_data.astype(
        {
            'Instrument S/N': 'int',
            'ZONE': 'int',
            'GMT DIFF.': 'float',
            'Gref': 'float',
            'Gcal1': 'float',
            'TiltxS': 'float',
            'TiltyS': 'float',
            'TiltxO': 'float',
            'TiltyO': 'float',
            'Tempco': 'float',
            'Drift': 'float',
            # 'DriftTime Start': [],
            # 'DriftDate Start': [],
            # 'Tide Correction': [],
            # 'Cont. Tilt': [],
            # 'Auto Rejection': [],
            # 'Terrain Corr.': [],
            # 'Seismic Filter': [],
            # 'Raw Data': [],
            'LINE': 'int',
            # 'STATION': [],
            'ALT.': 'float',
            'GRAV.': 'float',
            'SD.': 'float',
            'TILTX': 'float',
            'TILTY': 'float',
            'TEMP': 'float',
            'TIDE': 'float',
            'DUR': 'int',
            'REJ': 'int',
            # 'TIME': [],
            # 'DEC.TIME+DATE': [],
            'TERRAIN': 'float',
            # 'DATE': [],
        }, errors='ignore'
    )

    cg_data['Created'] = cg_data.apply(lambda x: dt.strptime(' '.join([x['Date'], x['Time']]), '%Y/ %m/%d %H %M %S'), axis=1)
    cg_data['Date_time'] = cg_data.apply(lambda x: dt.strptime(' '.join([x['DATE'], x['TIME']]), '%Y/%m/%d %H:%M:%S'), axis=1)

    return cg_data

def cg5_to_cg6_converter(cg5_data):

    cg6_data = pd.DataFrame()

    cg6_data['Survey Name'] = cg5_data['Survey name']
    cg6_data['Instrument Serial Number'] = cg5_data['Instrument S/N']
    cg6_data['Created'] = cg5_data['Created']
    cg6_data['Operator'] = cg5_data['Operator']
    cg6_data['Gcal1 [mGal]'] = cg5_data['Gcal1']
    cg6_data['Goff [ADU]'] = None
    cg6_data['Gref [mGal]'] = cg5_data['Gref']
    cg6_data['X Scale [arc-sec/ADU]'] = None
    cg6_data['Y Scale [arc-sec/ADU]'] = None
    cg6_data['X Offset [ADU]'] = None
    cg6_data['Y Offset [ADU]'] = None
    cg6_data['Temperature Coefficient [mGal/mK]'] = None
    cg6_data['Temperature Scale [mK/ADU]'] = None
    cg6_data['Drift Rate [mGal/day]'] = cg5_data['Drift']
    cg6_data['Drift Zero Time'] = cg5_data.apply(lambda x: dt.strptime(' '.join([x['DriftDate Start'], x['DriftTime Start']]), '%Y/%m/%d %H %M %S'), axis=1)
    cg6_data['Firmware Version'] = None
    cg6_data['Station'] = cg5_data['STATION']
    cg6_data['Date'] = cg5_data['DATE']
    cg6_data['Time'] = cg5_data['TIME']
    cg6_data['CorrGrav'] = cg5_data['GRAV.']
    cg6_data['Line'] = cg5_data['LINE'].astype('float').astype('int')
    cg6_data['StdDev'] = cg5_data['SD.']
    cg6_data['StdErr'] = None
    cg6_data['RawGrav'] = None
    cg6_data['X'] = cg5_data['TILTX']
    cg6_data['Y'] = cg5_data['TILTY']
    cg6_data['SensorTemp'] = cg5_data['TEMP']
    cg6_data['TideCorr'] = cg5_data['TIDE']
    cg6_data['TiltCorr'] = None
    cg6_data['TempCorr'] = None
    cg6_data['DriftCorr'] = None
    cg6_data['MeasurDur'] = cg5_data['DUR']
    cg6_data['InstrHeight'] = 0.0
    cg6_data['LatUser'] = None
    cg6_data['LonUser'] = None
    cg6_data['ElevUser'] = None
    cg6_data['LatGPS'] = None
    cg6_data['LonGPS'] = None
    cg6_data['ElevGPS'] = None
    cg6_data['Corrections[drift-temp-na-tide-tilt]'] = None
    cg6_data['MeterType'] = cg5_data['MeterType']
    cg6_data['DataFile'] = cg5_data['DataFile']
    cg6_data['date_time'] = cg5_data['Date_time']
 
    return cg6_data

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

    return cg_data

def read_scale_factors(calibration_files):
    
    ''' Get calibration factors from file(s) '''
    
    scale_factors = pd.DataFrame()

    for calibration_file in calibration_files:
        calibration_data = pd.read_csv(
            calibration_file.name,
            delimiter=' ',
            names = [
                'instrument_serial_number',
                'scale_factor',
                'scale_factor_std'
            ]
        )
        scale_factors = pd.concat([scale_factors, calibration_data], axis=0)

    scale_factors = scale_factors.reset_index()

    return scale_factors
    