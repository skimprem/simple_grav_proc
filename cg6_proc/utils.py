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
        'DATE': []
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
                    
                    for key, value in row.items():
                        rows[key].append(value)

    cg_data = pd.DataFrame(rows)

    exit()



def cg6_reader(data_files):
    meter_type = 'cg6'
    columns = [
        'survey_name',
        'instrument_serial_number',
        'created',
        'operator',
        'g_cal1',
        'g_off',
        'g_ref',
        'x_scale',
        'y_scale',
        'x_offset',
        'y_offset',
        'temperature_coefficient',
        'temperature_scale',
        'drift_rate',
        'drift_zero_time',
        'firmware_version',
        'station',
        'corr_grav',
        'line',
        'std_dev',
        'std_err',
        'raw_grav',
        'x',
        'y',
        'sensor_temp',
        'tide_corr',
        'tilt_corr',
        'temp_corr',
        'drift_corr',
        'measur_dur',
        'instr_height',
        'lat_user',
        'lon_user',
        'elev_user',
        'lat_gps',
        'lon_gps',
        'elev_gps',
        'corrections',
        'data_file',
        'meter_type'
    ]
    cg_data = pd.DataFrame(columns=columns)

    for data_file in data_files:
        if format_detect(data_file) != meter_type:
            raise ImportError(f'{data_file.name} data file must be in {meter_type.upper()} format')
        count = 0
        for line in data_file:
            count += 1
            if line[0] == '/':
                line = line[1:].strip()
                split_line = re.split(':\t', line)
                if line[:6] == 'Station':
                    continue
                elif not line:
                    continue
                match split_line[0]:
                    case 'CG-6 Survey':
                        continue
                    case 'CG-6 Calibration':
                        continue
                    case 'Survey Name':
                        survey_name = split_line[1]
                    case 'Instrument Serial Number':
                        instrument_serial_number = int(split_line[1])
                    case 'Created':
                        created = dt.strptime(split_line[1], "%Y-%m-%d %H:%M:%S")
                    case 'Operator':
                        operator = split_line[1]
                    case 'Gcal1 [mGal]':
                        gcal1 = float(split_line[1])
                    case 'Goff [ADU]':
                        goff = float(split_line[1])
                    case 'Gref [mGal]':
                        gref = float(split_line[1])
                    case 'X Scale [arc-sec/ADU]':
                        x_scale = float(split_line[1])
                    case 'Y Scale [arc-sec/ADU]':
                        y_scale = float(split_line[1])
                    case 'X Offset [ADU]':
                        x_offset = float(split_line[1])
                    case 'Y Offset [ADU]':
                        y_offset = float(split_line[1])
                    case 'Temperature Coefficient [mGal/mK]':
                        temperature_coefficient = float(split_line[1])
                    case 'Temperature Scale [mK/ADU]':
                        temperature_scale = float(split_line[1])
                    case 'Drift Rate [mGal/day]':
                        drift_rate = float(split_line[1])
                    case 'Drift Zero Time':
                        drift_zero_time = dt.strptime(
                            split_line[1],
                            "%Y-%m-%d %H:%M:%S"
                        )
                    case 'Firmware Version':
                        firmware_version = split_line[1]
            else:
                if not line.strip():
                    continue
                split_line = line.split()
                try:
                    float(split_line[3])
                    station, date_, time_, corrgrav, line_, stddev,\
                        stderr, rawgrav, x, y, sensortemp, tidecorr,\
                        tiltcorr, tempcorr, driftcorr, measurdur,\
                        instrheight, latuser, lonuser, elevuser, latgps,\
                        longps, elevgps, corrections = split_line
                except ValueError:
                    print(f'Warning: ValueError at line {count}')
                    continue
                date_time = dt.strptime(
                    date_ + 'T' + time_, "%Y-%m-%dT%H:%M:%S")
                corrgrav = float(corrgrav) * 1e3
                line_ = int(line_)
                stddev = float(stddev) * 1e3
                stderr = float(stderr) * 1e3
                rawgrav = float(rawgrav) * 1e3
                x = float(x)
                y = float(y)
                sensortemp = float(sensortemp)
                tidecorr = float(tidecorr) * 1e3
                tiltcorr = float(tiltcorr) * 1e3
                tempcorr = float(tempcorr) * 1e3
                driftcorr = float(driftcorr) * 1e3
                measurdur = float(measurdur)
                instrheight = float(instrheight) * 1e3
                latuser = float(latuser)
                lonuser = float(lonuser)
                elevuser = float(elevuser)

                try:
                    latgps = float(latgps)
                except ValueError:
                    latgps = None
                try:
                    longps = float(longps)
                except ValueError:
                    longps = None
                try:
                    elevgps = float(elevgps)
                except ValueError:
                    elevgps = None

                cg_data.loc[date_time] = [
                    survey_name,
                    instrument_serial_number,
                    created,
                    operator,
                    gcal1,
                    goff,
                    gref,
                    x_scale,
                    y_scale,
                    x_offset,
                    y_offset,
                    temperature_coefficient,
                    temperature_scale,
                    drift_rate,
                    drift_zero_time,
                    firmware_version,
                    station,
                    corrgrav,
                    line_,
                    stddev,
                    stderr,
                    rawgrav,
                    x,
                    y,
                    sensortemp,
                    tidecorr,
                    tiltcorr,
                    tempcorr,
                    driftcorr,
                    measurdur,
                    instrheight,
                    latuser,
                    lonuser,
                    elevuser,
                    latgps,
                    longps,
                    elevgps,
                    corrections,
                    data_file.name,
                    meter_type
                ]
        data_file.close()

    return cg_data


def get_readings(cg_data):
    ''' Get mean values of signals of readings '''
    readings = pd.DataFrame(
        columns=[
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
            'lon_user'
        ]
    )
    for line in cg_data.line.unique():
        line_data = cg_data[cg_data.line == line]
        trigger = False
        count = 0
        first_index = line_data.index[0]
        for index, row in line_data.iterrows():
            if index == line_data.index[-1]:
                station_mean = line_data.corr_grav.loc[
                    first_index:index].mean()
                height_mean = line_data.instr_height.loc[
                    first_index:index].mean()
                mean_time = first_index + (index - first_index) / 2
                readings.loc[mean_time] = [
                    row.created,
                    row.survey_name,
                    row.operator,
                    row.instrument_serial_number,
                    height_mean,
                    row.line,
                    row.station,
                    station_mean,
                    row.data_file,
                    row.lat_user,
                    row.lon_user]
                break
            if row.station == line_data.station.iloc[count + 1]:
                count += 1
                if not trigger:
                    trigger = True
                    first_index = index
            else:
                trigger = False
                station_mean = line_data.corr_grav.loc[
                    first_index:index].mean()
                height_mean = line_data.instr_height.loc[
                    first_index:index].mean()
                mean_time = first_index + (index - first_index) / 2
                readings.loc[mean_time] = [
                    row.created,
                    row.survey_name,
                    row.operator,
                    row.instrument_serial_number,
                    height_mean,
                    row.line,
                    row.station,
                    station_mean,
                    row.data_file,
                    row.lat_user,
                    row.lon_user]
                count += 1

    for station in readings.station.unique():
        station_readings = readings[readings.station == station]
        first_lat = station_readings.lat_user[0]
        first_lon = station_readings.lon_user[0]
        for index, row in station_readings.iterrows():
            readings.loc[
                index, ['lat_user', 'lon_user']] = [first_lat, first_lon]

    return readings


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
        'lon_user_to'
    ])
    count = 0
    for line in readings.line.unique():
        line_readings = readings[readings.line == line]
        loops = []
        for index, row in line_readings.iterrows():
            if len(loops) == 0:
                loops.append({
                    'date_time': index,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user
                })
                continue
            if row.station == loops[0]['station']:
                begin_index = loops[0]['date_time']
                factor = (
                    row.corr_grav - line_readings.corr_grav.loc[begin_index]
                ) / (
                    dt.timestamp(index) - dt.timestamp(begin_index)
                )
                for reading in loops[1:]:
                    correction = factor * (
                        dt.timestamp(reading['date_time'])
                        - dt.timestamp(begin_index)
                    )
                    tie = reading['corr_grav'] -\
                        loops[0]['corr_grav'] + correction
                    level_from = loops[0]['instr_height']
                    level_to = reading['instr_height']
                    date = reading['created']
                    date_from = begin_index
                    date_to = reading['date_time']
                    site = reading['survey_name']
                    operator = reading['operator']
                    meter = reading['instrument_serial_number']
                    data_file = reading['data_file']
                    lat_user = reading['lat_user']
                    lon_user = reading['lon_user']
                    ties.loc[count] = [
                        date_from,
                        date_to,
                        date,
                        site,
                        operator,
                        meter,
                        level_to,
                        level_from,
                        row.line,
                        row.station,
                        reading['station'],
                        tie,
                        data_file,
                        row.lat_user,
                        lat_user,
                        row.lon_user,
                        lon_user
                    ]
                    count += 1
                loops.pop(0)
                loops.append({
                    'date_time': index,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user
                })
            else:
                loops.append({
                    'date_time': index,
                    'created': row.created,
                    'survey_name': row.survey_name,
                    'operator': row.operator,
                    'station': row.station,
                    'corr_grav': row.corr_grav,
                    'instr_height': row.instr_height,
                    'instrument_serial_number': row.instrument_serial_number,
                    'data_file': row.data_file,
                    'lat_user': row.lat_user,
                    'lon_user': row.lon_user
                })
    return ties


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
                    tie_row['lat_user_to']
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
        'date_time'
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
            'date_to': 'last'
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
            'date_time'
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
        tie.date_time
    ]
    return reverse


def get_report(means):
    ''' Get table report of means '''
    report = 'The mean ties between the stations:'
    columns = [
        'station_from',
        'station_to',
        'date_time',
        'survey_name',
        'operator',
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
    report = f'{report}\n\n{means_table}'
    headers = ['Cicles', 'Sum (uGals)']
    sums_table = get_ties_sum(means).to_markdown(
        index=False,
        headers=headers,
        tablefmt="simple",
        floatfmt=".2f")
    report = f'{report}\n\nSum of the ties:\n\n{sums_table}'
    return report


def make_vgfit_input(means, filename):
    ''' Make CSV file for vg_fit utilite '''
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
    means_to_vgfit = means[columns]
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
    return means_to_vgfit


def get_residuals_plot(raw, readings, ties):
    ''' Get plot of residuals '''
    for _, tie_row in ties.iterrows():
        tie_readings = raw[raw.line == tie_row.line]
        first_reading = readings[readings.line == tie_row.line].corr_grav[0]
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

    survey_names = ', '.join(str(survey_name) for survey_name in readings.survey_name.unique())
    delta_time = readings.index[-1] - readings.index[0]
    if delta_time < td(hours=24):
        date_formatter = DateFormatter('%H:%M')
    elif delta_time > td(days=2):
        date_formatter = DateFormatter('%b %d')
    else:
        date_formatter = DateFormatter('%b %d %H:%M')
    sns.set(style="whitegrid")
    plt.xlabel('Date & Time')
    plt.ylabel('Residuals [uGals]')
    plt.title(f'Residuals of {survey_names} surveys')
    sns.scatterplot(
        raw,
        x=raw.index,
        y='residuals',
        hue='station').xaxis.set_major_formatter(date_formatter)
    plt.legend(title='Stations')

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
        'lat_user'
    ]
    group = [
        'survey_name',
        'station'
    ]
    agg = {
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
