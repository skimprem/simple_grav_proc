'''
Set of utilites for relative gravity processing
'''

from datetime import datetime as dt
from datetime import timedelta as td
from time import time as tm
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import networkx as nx


def format_detect(data_file):
    for line in data_file:
        line = line.strip()
        if not line:
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


def make_frame_to_proc(cg_data):
    ''' Make a data frame to processing (only needed columns to be selected) '''
    match cg_data.iloc[0].MeterType:
        case 'CG6':
            data = cg_data[
                [
                    'date_time',
                    'Created',
                    'Survey Name',
                    'Operator',
                    'Instrument Serial Number',
                    'InstrHeight',
                    'Line',
                    'Station',
                    'CorrGrav',
                    'DataFile',
                    'LatUser',
                    'LonUser',
                    'MeterType'
                ]
            ]

            headers = [
                'date_time',
                'created',
                'survey_name',
                'operator',
                'instrument_serial_number',
                'instr_height',
                'line',
                'station',
                'corr_grav',
                'data_file',
                'lat_user',
                'lon_user',
                'meter_type'
            ]

            data.columns = headers
            for index, row in data.iterrows():
                data.loc[index, 'corr_grav'] = row.corr_grav * 1e3
                data.loc[index, 'instr_height'] = row.instr_height * 1e3
            return data
        case 'CG5':
            pass


def get_meters_readings(cg_data):
    ''' Get mean values of signals of readings from different meters '''
    readings = pd.DataFrame(
        columns=[
            'date_time',
            'created',
            'survey_name',
            'operator',
            'instrument_serial_number',
            'instr_height',
            'line',
            'station',
            'corr_grav',
            'data_file',
            'lat_user',
            'lon_user',
            'meter_type'
        ]
    )
    for meter in cg_data.instrument_serial_number.unique():
        meter_data = cg_data[cg_data.instrument_serial_number == meter]
        meter_readings = get_readings(meter_data)
        readings = pd.concat([readings, meter_readings])
    return readings


def get_readings(cg_data):
    ''' Get mean values of signals of readings '''
    readings = pd.DataFrame(
        columns=[
            'date_time',
            'created',
            'survey_name',
            'operator',
            'instrument_serial_number',
            'instr_height',
            'line',
            'station',
            'corr_grav',
            'data_file',
            'lat_user',
            'lon_user',
            'meter_type'
        ]
    )
    reading_index = 0
    for line in cg_data.line.unique():
        line_data = cg_data[cg_data.line == line]
        trigger = False
        count = 0
        first_date_time = line_data.iloc[0].date_time
        first_index = line_data.iloc[0].index
        for index, row in line_data.iterrows():
            if index == line_data.index[-1]:
                readings.loc[reading_index] = [
                    first_date_time + (row.date_time - first_date_time) / 2,
                    row.created,
                    row.survey_name,
                    row.operator,
                    row.instrument_serial_number,
                    line_data.loc[first_index:index].instr_height.mean(),
                    row.line,
                    row.station,
                    line_data.loc[first_index:index].corr_grav.mean(),
                    row.data_file,
                    line_data.loc[first_index:index].lat_user.mean(),
                    line_data.loc[first_index:index].lon_user.mean(),
                    row.meter_type
                ]
                reading_index += 1
                break
            if row.station == line_data.station.iloc[count + 1]:
                count += 1
                if not trigger:
                    trigger = True
                    first_date_time = row.date_time
                    first_index = index
            else:
                trigger = False
                readings.loc[reading_index] = [
                    first_date_time + (row.date_time - first_date_time) / 2,
                    row.created,
                    row.survey_name,
                    row.operator,
                    row.instrument_serial_number,
                    line_data.loc[first_index:index].instr_height.mean(),
                    row.line,
                    row.station,
                    line_data.loc[first_index:index].corr_grav.mean(),
                    row.data_file,
                    line_data.loc[first_index:index].lat_user.mean(),
                    line_data.loc[first_index:index].lon_user.mean(),
                    row.meter_type
                ]
                count += 1
                reading_index += 1

    for station in readings.station.unique():
        station_readings = readings[readings.station == station]
        first_lat = station_readings.iloc[0].lat_user
        first_lon = station_readings.iloc[0].lon_user
        for index, row in station_readings.iterrows():
            readings.loc[
                index, ['lat_user', 'lon_user']] = [first_lat, first_lon]
    return readings


def get_meters_ties(readings):
    ''' Get ties from meters '''
    ties = pd.DataFrame(columns=[
        'date_from',
        'date_to',
        'created',
        'survey_name',
        'operator',
        'instrument_serial_number',
        'instr_height_from',
        'instr_height_to',
        'line',
        'station_from',
        'station_to',
        'tie',
        'data_file',
        'lat_user_from',
        'lat_user_to',
        'lon_user_from',
        'lon_user_to',
        'meter_type'
    ])
    for meter in readings.instrument_serial_number.unique():
        meter_readings = readings[readings.instrument_serial_number == meter]
        meter_ties = get_ties(meter_readings)
        ties = pd.concat([ties, meter_ties])
    return ties


def get_ties(readings):
    ''' Get ties '''
    ties = pd.DataFrame(columns=[
        'date_from',
        'date_to',
        'created',
        'survey_name',
        'operator',
        'instrument_serial_number',
        'instr_height_from',
        'instr_height_to',
        'line',
        'station_from',
        'station_to',
        'tie',
        'data_file',
        'lat_user_from',
        'lat_user_to',
        'lon_user_from',
        'lon_user_to',
        'meter_type'
    ])
    count = 0
    for line in readings.line.unique():
        line_readings = readings[readings.line == line]
        loops = []
        for index, row in line_readings.iterrows():
            if len(loops) == 0:
                loops.append({
                    'index': index,
                    'date_time': row.date_time,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user,
                    'meter_type': row.meter_type
                })
                continue
            if row.station == loops[0]['station']:
                begin_date_time = loops[0]['date_time']
                begin_index = loops[0]['index']
                factor = (
                    row.corr_grav - line_readings.loc[begin_index].corr_grav
                ) / (
                    dt.timestamp(row.date_time) - dt.timestamp(begin_date_time)
                )
                for reading in loops[1:]:
                    correction = factor * (
                        dt.timestamp(reading['date_time'])
                        - dt.timestamp(begin_date_time)
                    )
                    ties.loc[count] = [
                        begin_date_time,
                        reading['date_time'],
                        reading['created'],
                        reading['survey_name'],
                        reading['operator'],
                        reading['instrument_serial_number'],
                        reading['instr_height'],
                        loops[0]['instr_height'],
                        row.line,
                        row.station,
                        reading['station'],
                        reading['corr_grav'] - loops[0]['corr_grav'] + correction,
                        reading['data_file'],
                        row.lat_user,
                        reading['lat_user'],
                        row.lon_user,
                        reading['lon_user'],
                        reading['meter_type']
                    ]
                    count += 1
                loops.pop(0)
                loops.append({
                    'index': index,
                    'date_time': row.date_time,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user,
                    'meter_type': row.meter_type
                })
            else:
                loops.append({
                    'index': index,
                    'date_time': row.date_time,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user,
                    'meter_type': row.meter_type
                })
    return ties

    
def get_meters_mean_ties(ties):
    mean_ties = pd.DataFrame(columns=[
        'station_from',
        'station_to',
        'created',
        'operator',
        'instrument_serial_number',
        'line',
        'instr_height_from',
        'instr_height_to',
        'tie',
        'std',
        'data_file',
        'lat_user_from',
        'lat_user_to',
        'lon_user_from',
        'lon_user_to',
        'date_time',
        'meter_type'

    ])
    for meter in ties.instrument_serial_number.unique():
        meter_ties = ties[ties.instrument_serial_number == meter]
        meter_mean_ties = get_mean_ties(meter_ties)
        mean_ties = pd.concat([mean_ties, meter_mean_ties])
    return mean_ties


def get_mean_ties(ties):
    ''' Get mean values of ties '''
    for _, row in ties.iterrows():
        from_station = row['station_from']
        to_station = row['station_to']
        for tie_index, tie_row in ties.iterrows():
            station_from = tie_row['station_from']
            station_to = tie_row['station_to']
            if station_from == to_station and station_to == from_station:
                ties.loc[tie_index] = [
                    tie_row['date_from'],
                    tie_row['date_to'],
                    tie_row['created'],
                    tie_row['survey_name'],
                    tie_row['operator'],
                    tie_row['instrument_serial_number'],
                    row['instr_height_to'],
                    row['instr_height_from'],
                    tie_row['line'],
                    from_station,
                    to_station,
                    -tie_row['tie'],
                    tie_row['data_file'],
                    tie_row['lat_user_from'],
                    tie_row['lat_user_to'],
                    tie_row['lat_user_from'],
                    tie_row['lat_user_to'],
                    tie_row['meter_type']
                ]

    result = pd.DataFrame(columns=[
        'station_from',
        'station_to',
        'created',
        'station',
        'operator',
        'instrument_serial_number',
        'line',
        'instr_height_from',
        'instr_height_to',
        'tie',
        'std',
        'data_file',
        'lat_user_from',
        'lat_user_to',
        'lon_user_from',
        'lon_user_to',
        'date_time',
        'meter_type'
    ])

    means = []
    for line in ties.line.unique():
        line_ties = ties[ties.line == int(line)]
        for index_line_tie, _ in line_ties.iterrows():
            line_ties.loc[index_line_tie, 'date_to'] =\
                dt.date(line_ties.loc[index_line_tie, 'date_to'])
        group_mean = line_ties.groupby(
            ['station_from', 'station_to'], as_index=False)
        mean = group_mean.agg({
            'created': 'last',
            'survey_name': 'last',
            'operator': 'last',
            'instrument_serial_number': 'last',
            'line': 'last',
            'instr_height_from': 'mean',
            'instr_height_to': 'mean',
            'tie': ['mean', 'std'],
            'data_file': 'last',
            'lat_user_from': 'mean',
            'lat_user_to': 'mean',
            'lon_user_from': 'mean',
            'lon_user_to': 'mean',
            'date_to': 'last',
            'meter_type': 'last'
        })
        mean.columns = [
            'station_from',
            'station_to',
            'created',
            'survey_name',
            'operator',
            'instrument_serial_number',
            'line',
            'instr_height_from',
            'instr_height_to',
            'tie',
            'std',
            'data_file',
            'lat_user_from',
            'lat_user_to',
            'lon_user_from',
            'lon_user_to',
            'date_time',
            'meter_type'
        ]
        means.append(mean)
    result = pd.concat(means, ignore_index=True)
    return result


def get_ties_sum(ties):
    ''' Get sum of ties '''
    nodes = list(ties.station_from)
    nodes.extend(list(ties.station_to))
    nodes = list(set(nodes))
    edges = []
    for _, row in ties.iterrows():
        edges.append((row.station_from, row.station_to))

    ties_graph = nx.Graph()
    ties_graph.add_nodes_from(nodes)
    ties_graph.add_edges_from(edges)

    cicles_sum = {
        'cicle': [],
        'sum': []
    }
    for cicle in nx.simple_cycles(ties_graph):
        cicle.append(cicle[0])
        cicle_ties = []
        for station_index in range(len(cicle)-1):
            line_ties = []
            for _, tie_row in ties.iterrows():
                station_from = cicle[station_index]
                station_to = cicle[station_index+1]
                if station_from == tie_row.station_from and\
                        station_to == tie_row.station_to:
                    line_ties.append(tie_row.tie)
                elif station_from == tie_row.station_to and\
                        station_to == tie_row.station_from:
                    line_ties.append(-tie_row.tie)
                else:
                    continue
            line_tie = sum(line_ties) / len(line_ties)
            cicle_ties.append(line_tie)
        cicle.pop()
        cicle_line = '-'.join([str(station) for station in cicle])
        cicles_sum['cicle'].append(cicle_line)
        cicles_sum['sum'].append(sum(cicle_ties))

    cicles = pd.DataFrame(cicles_sum)

    return cicles


def reverse_tie(tie):
    ''' Reverse of tie (from -> to, to -> from, tie = - tie) '''
    reverse = [
        tie.station_to,
        tie.station_from,
        tie.created,
        tie.survey_name,
        tie.operator,
        tie.instrument_serial_number,
        tie.line,
        tie.instr_height_to,
        tie.instr_height_from,
        -tie.tie,
        tie['std'],
        tie.data_file,
        tie.lat_user_from,
        tie.lat_user_to,
        tie.lon_user_from,
        tie.lon_user_to,
        tie.date_time,
        tie.meter_type
    ]
    return reverse

def get_meters_report(means):
    report = ''    
    for meter in means.instrument_serial_number.unique():
        meter_means = means[means.instrument_serial_number == meter]
        meter_report = get_report(meter_means)
        report = report + meter_report
    return report


def get_report(means):
    ''' Get table report of means '''
    report = f'The mean ties between the stations for meter #{means.iloc[0].instrument_serial_number}:'
    columns = [
        'station_from',
        'station_to',
        'date_time',
        'survey_name',
        'operator',
        'meter_type',
        'instrument_serial_number',
        'line',
        'instr_height_from',
        'instr_height_to',
        'tie',
        'std'
    ]
    headers = [
        'From',
        'To',
        'Date',
        'Survey',
        'Operator',
        'Meter',
        'S/N',
        'Line',
        'Height From (mm)',
        'Height To (mm)',
        'Tie (uGals)',
        'SDev (uGals)'
    ]
    means = means.replace(np.nan, None)
    means_table = means[columns].to_markdown(
        index=False,
        headers=headers,
        tablefmt="simple",
        floatfmt=".1f")
    report = f'{report}\n{means_table}\n\n'
    ties_sum = get_ties_sum(means)
    if len(ties_sum):
        headers = ['Cicles', 'Sum (uGals)']
        sums_table = ties_sum.to_markdown(
            index=False,
            headers=headers,
            tablefmt="simple",
            floatfmt=".2f")
        report = f'{report}Sum of the ties:\n{sums_table}\n\n'
    return report


def make_vgfit_input(means):
    ''' Make CSV file for vg_fit utilite '''
    for meter in means.instrument_serial_number.unique():
        meter_means = means[means.instrument_serial_number == meter]
        filename = meter_means.iloc[0].survey_name+'_'+str(meter)+'.csv'
        columns = [
            'date_time',
            'survey_name',
            'operator',
            'instrument_serial_number',
            'line',
            'instr_height_from',
            'instr_height_to',
            'tie',
            'std',
            'data_file'
        ]
        means_to_vgfit = meter_means[columns]
        means_to_vgfit.columns = [
            'date',
            'station',
            'observer',
            'gravimeter',
            'runn',
            'level_1',
            'level_2',
            'delta_g',
            'std',
            'source'
        ]
        means_to_vgfit.to_csv(filename, index=False)
    return


def get_residuals_plot(raw, readings, ties):
    ''' Get plot of residuals '''
    meters = ties.instrument_serial_number.unique()
    for meter in meters:
        meter_raw = raw[raw.instrument_serial_number == meter]
        meter_readings = readings[readings.instrument_serial_number == meter]
        meter_ties = ties[ties.instrument_serial_number == meter]
        for _, tie_row in meter_ties.iterrows():
            tie_readings = meter_raw[meter_raw.line == tie_row.line]
            first_reading = meter_readings[meter_readings.line == tie_row.line].iloc[0].corr_grav
            tie_station = tie_row.station_to
            for reading_index, reading_row in tie_readings.iterrows():
                if reading_row.station == tie_station:
                    raw.loc[
                        reading_index,
                        ['residuals']] = reading_row.corr_grav\
                            - first_reading - tie_row.tie
                else:
                    raw.loc[
                        reading_index,
                        ['residuals']] = reading_row.corr_grav - first_reading

    # delta_time = readings.iloc[-1].date_time - readings.iloc[0].date_time
    # if delta_time < td(hours=24):
    #     date_formatter = DateFormatter('%H:%M')
    # elif delta_time > td(days=2):
    #     date_formatter = DateFormatter('%b %d')
    # else:
    #     date_formatter = DateFormatter('%b %d %H:%M')
    meter_type = raw.iloc[0].meter_type
    with sns.axes_style("whitegrid"):
        plots = sns.FacetGrid(
            raw,
            col='instrument_serial_number',
            hue='station',
            col_wrap=1,
            aspect=4,
            margin_titles=True,
            sharey=False,
            sharex=False
        )
    plots.map(
        sns.scatterplot,
        'date_time',
        'residuals')
    plots.set_axis_labels('Date & Time [UTC]', 'Residuals [uGals]')
    plots.set_titles('Residuals of '+meter_type+' {col_name}')
    plots.add_legend(title='Stations')
    # plots.axes[0].xaxis.set_major_formatter(date_formatter)

    return raw


def get_map(readings):
    ''' Get map of ties scheme '''
    columns = ['station', 'lat_user', 'lon_user']
    group = ['station']
    stations = readings[columns].groupby(group).mean()
    stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations.lon_user,
            stations.lat_user),
        crs='epsg:4326')
    columns = [
        'survey_name',
        'instrument_serial_number',
        'created',
        'operator',
        'station',
        'data_file',
        'lon_user',
        'lat_user',
        'meter_type'
    ]
    group = [
        'survey_name',
        'station'
    ]
    agg = {
        'meter_type': 'last',
        'instrument_serial_number': 'last',
        'created': 'last',
        'operator': 'last',
        'data_file': 'last',
        'lon_user': 'mean',
        'lat_user': 'mean'
    }
    lines = readings[columns].groupby(group).agg(agg)
    lines = gpd.GeoDataFrame(
        lines,
        geometry=gpd.points_from_xy(
            lines.lon_user,
            lines.lat_user),
        crs='epsg:4326')
    lines = lines.sort_values(
        by=['station']).groupby(
            ['survey_name'])['geometry'].apply(
                lambda x: LineString(x.tolist()))
    lines = gpd.GeoDataFrame(lines, geometry='geometry', crs='epsg:4326')

    stations.plot()
    ties_map = lines.explore(
        legend=True
    )
    ties_map = stations.explore(
        m=ties_map,
        color='red'
    )

    return ties_map
