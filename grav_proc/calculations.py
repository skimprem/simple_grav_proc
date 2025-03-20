import pandas as pd
from datetime import datetime as dt
# from time import time as tm
import numpy as np
import networkx as nx
import statsmodels.api as sm


def replace_lat(lat_user, lat_gps):
    if lat_gps == '--':
        result = lat_user
    else:
        result = lat_gps
    return float(result)

def replace_lon(lon_user, lon_gps):
    if lon_gps == '--':
        result = lon_user
    else:
        result = lon_gps
    return float(result)


def make_frame_to_proc(cg_data):
    ''' Make a data frame to processing (only needed columns to be selected) '''


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
        'std_err',
        'data_file',
        'lat',
        'lon',
        'meter_type',
    ]

    data = cg_data

    data['CorrGrav'] = data['CorrGrav'] * 1e3
    data['StdErr']  = data['StdErr'] * 1e3
    data['InstrHeight'] = data['InstrHeight'] * 1e3

    group_by_meter_station = data.groupby(['Instrument Serial Number', 'Station'])
    for meter_station, grouped_by_meter_station in group_by_meter_station:
        meter, station = meter_station
        indices = grouped_by_meter_station.index

        grouped_by_meter_station['LatGPS'] = grouped_by_meter_station.apply(lambda x: replace_lat(x['LatUser'], x['LatGPS']), axis=1)
        grouped_by_meter_station['LonGPS'] = grouped_by_meter_station.apply(lambda x: replace_lat(x['LonUser'], x['LonGPS']), axis=1)
        
        lat = grouped_by_meter_station['LatGPS'].mean()
        lat_std = grouped_by_meter_station['LatGPS'].std()
        lon = grouped_by_meter_station['LonGPS'].mean()
        lon_std = grouped_by_meter_station['LonGPS'].std()

        if lat_std > 0.001 or lon_std > 0.001:
            print(f'WARNING: {meter} {station}')
            print('    lat_std = ', lat_std, ', lon_std = ', lon_std)

        data.loc[indices, 'lat'] = lat
        data.loc[indices, 'lon'] = lon
    
    data = data[
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
            'StdErr',
            'DataFile',
            'lat',
            'lon',
            'MeterType',
        ]
    ]

    data.columns = headers

    return data


def get_meters_readings(cg_data):
    ''' Get mean values of signals of readings from different meters '''

    readings = pd.DataFrame()
    group_by_meters = cg_data.groupby('instrument_serial_number')
    for meter, meter_data in group_by_meters:
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
            'std_err',
            'data_file',
            'lat_user',
            'lon_user',
            'meter_type',
        ]
    )
    # readings['date_time'].to_datetime
    reading_index = 0
    group_by_line = cg_data.groupby('line')
    for line, line_data in group_by_line:
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
                    # line_data.loc[first_index:index].apply(lambda x: weighted_avg_and_std(x.corr_grav, x.std_err**-2)[0], axis=1),
                    line_data.loc[first_index:index].corr_grav.mean(),
                    # line_data.loc[first_index:index].apply(lambda x: weighted_avg_and_std(x.corr_grav, x.std_err**-2)[1], axis=1),
                    line_data.loc[first_index:index].corr_grav.sem(),

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
                    # line_data.loc[first_index:index].apply(lambda x: weighted_avg_and_std(x.corr_grav, x.std_err**-2)[0], axis=1),
                    line_data.loc[first_index:index].corr_grav.mean(),
                    # line_data.loc[first_index:index].apply(lambda x: weighted_avg_and_std(x.corr_grav, x.std_err**-2)[1], axis=1),
                    line_data.loc[first_index:index].corr_grav.sem(),
                    row.data_file,
                    line_data.loc[first_index:index].lat_user.mean(),
                    line_data.loc[first_index:index].lon_user.mean(),
                    row.meter_type
                ]
                count += 1
                reading_index += 1

    group_by_station = readings.groupby('station')
    
    for station, station_readings in group_by_station:
        mean_lat = station_readings.lat_user.mean()
        mean_lon = station_readings.lon_user.mean()
        readings.loc[readings.station == station, 'lat_user'] = mean_lat
        readings.loc[readings.station == station, 'lon_user'] = mean_lon

    return readings

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def get_meters_ties(readings):
    ''' Get ties from meters '''
    ties = pd.DataFrame()
    
    group_by_meters = readings.groupby('instrument_serial_number')
    
    for meter, meter_readings in group_by_meters:
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
    group_by_line = readings.groupby('line')
    for line, line_readings in group_by_line:
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
    mean_ties = pd.DataFrame()
    group_by_meters = ties.groupby('instrument_serial_number')
    for meter, meter_ties in group_by_meters:
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
        'err',
        'data_file',
        'lat_user_from',
        'lat_user_to',
        'lon_user_from',
        'lon_user_to',
        'date_time',
        'meter_type'
    ])

    means = []
    group_by_line = ties.groupby('line')
    for line, line_ties in group_by_line:
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
            'tie': ['mean', 'sem'],
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
            'err',
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
    meter = f'{ties.iloc[0].meter_type} #{str(ties.iloc[0].instrument_serial_number)}'
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
        'meter': [],
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
        cicles_sum['meter'].append(meter)
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
        tie['err'],
        tie.data_file,
        tie.lat_user_from,
        tie.lat_user_to,
        tie.lon_user_from,
        tie.lon_user_to,
        tie.date_time,
        tie.meter_type
    ]
    return reverse


def to_minutes(value):
    return value.timestamp() / 60

def to_days(value):
    return value.timestamp() / 60 / 60 / 24

def to_seconds(value):
    return value.timestamp()

def free_grav_fit(stations, gravity, date_time, fix_station, std=None, max_degree=2, method='WLS'):
    
    observation_matrix = pd.get_dummies(stations).drop(fix_station, axis=1)

    defined_stations = observation_matrix.columns

    # date_time = date_time - date_time.iloc[0]
    time_matrix = np.vander(date_time, max_degree)
    
    design_matrix = np.hstack((observation_matrix, time_matrix))

    # model = sm.RLM(input_grav, design_matrix)
    match method:
        case 'RLM':
            model = sm.RLM(gravity, design_matrix)
        case 'WLS':
            if std is not None:
                model = sm.WLS(gravity, design_matrix, weights=std**-2)
            else:
                model = sm.WLS(gravity, design_matrix)

    result = model.fit()

    ties = pd.DataFrame()

    for index, station in enumerate(defined_stations):
        ties = pd.concat(
            [
                ties,
                pd.DataFrame(
                    {
                        'station_from': fix_station,
                        'station_to': station,
                        'tie': result.params.iloc[index],
                        'err': result.bse.iloc[index],
                        # 'line': f'{fix_station}-{station}'
                    }, index=[0]
                )
            ], ignore_index=True
        )
    
    return ties, result.resid


def drift_fitting(stations, grav, std_err, date_time, fix_station=None, max_degree=2):
    ''' drift_fitting'''
    
    readings = np.vstack(grav.array)
    readings = readings - readings[0]

    date_time = date_time - date_time.iloc[0]
    times = np.vstack(date_time.dt.seconds.array)

    desired_stations = stations.unique()
    if fix_station:
        desired_stations = desired_stations[desired_stations != fix_station]
    else:
        desired_stations = desired_stations[desired_stations != desired_stations[0]]
    
    if max_degree > 1:
        for degree in range(2, max_degree + 1):
            times = np.hstack((times, np.power(times, degree)))
    
    rows = []
    for station in stations:
        row = []
        for desired_station in desired_stations:
            value = 1 if desired_station == station else 0
            row.append(value)
        rows.append(row)
    station_grav = np.array(rows)
    
    ones = np.ones(shape=(readings.size, 1))
    
    design_matrix = np.concatenate((station_grav, times, ones), axis=1)
    # design_matrix = np.concatenate((times, ones), axis=1)

    # model = sm.OLS(readings, design_matrix)
    # model = sm.WLS(readings, design_matrix, weights=np.array(std_err)**-2)
    model = sm.RLM(readings, design_matrix)
    
    result = model.fit()

    return [(value, std) for value, std in zip(result.params, result.bse)]

def get_meter_ties_by_lines(readings):
    
    lines = {
        'station_from': [],
        'station_to': [],
        'created': [],
        'survey_name': [],
        'operator': [],
        'instrument_serial_number': [],
        'line': [],
        'instr_height_from': [],
        'instr_height_to': [],
        'tie': [],
        'err': [],
    }

    for meter, meter_data in readings.groupby('instrument_serial_number'):

        for line, line_data in meter_data.groupby('line'):
            stations = line_data.station.unique()
            first_station = stations[0]
            stations = stations[stations != first_station]
            instr_height_from = line_data[line_data.station == first_station].instr_height.mean()
            gravs = drift_fitting(
                stations=line_data.station,
                grav=line_data.corr_grav,
                std_err=line_data.std_err,
                date_time=line_data.date_time,
                max_degree=2
            )[:len(stations)]
            for station, grav in zip(stations, gravs):
                lines['station_from'].append(first_station)
                lines['station_to'].append(station)
                lines['created'].append(line_data[line_data.station == station].iloc[0].created)
                lines['survey_name'].append(line_data[line_data.station == station].iloc[0].survey_name)
                lines['operator'].append(line_data[line_data.station == station].iloc[0].operator)
                lines['instrument_serial_number'].append(meter)
                lines['line'].append(line)
                lines['instr_height_from'].append(instr_height_from)
                lines['instr_height_to'].append(line_data[line_data.station == station].instr_height.mean())
                lines['tie'].append(grav[0])
                lines['err'].append(grav[1])
        
    return pd.DataFrame(lines)

def get_meter_ties_all(readings):
    
    lines = {}

    lines['station_from'] = []
    lines['station_to'] = []
    lines['created'] = []
    lines['survey_name'] = []
    lines['operator'] = []
    lines['instrument_serial_number'] = []
    lines['instr_height_from'] = []
    lines['instr_height_to'] = []
    lines['tie'] = []
    lines['err'] = []

    stations = readings.station.unique()
    first_station = stations[0]
    stations = stations[stations != first_station]
    instr_height_from = readings[readings.station == first_station].instr_height.mean()
    result = drift_fitting(
        stations=readings.station,
        grav=readings.corr_grav,
        std_err=readings.std_err,
        date_time=readings.date_time,
        max_degree=2
    )
    gravs = result[:len(stations)]
    for station, grav in zip(stations, gravs):
        lines['station_from'].append(first_station)
        lines['station_to'].append(station)
        lines['created'].append(readings[readings.station == station].iloc[0].created)
        lines['survey_name'].append(readings[readings.station == station].iloc[0].survey_name)
        lines['operator'].append(readings[readings.station == station].iloc[0].operator)
        lines['instrument_serial_number'].append(readings[readings.station == station].iloc[0].instrument_serial_number)
        lines['instr_height_from'].append(instr_height_from)
        lines['instr_height_to'].append(readings[readings.station == station].instr_height.mean())
        lines['tie'].append(grav[0])
        lines['err'].append(grav[1])

    return pd.DataFrame(lines)

def fit_by_meter_created(raw_data, anchor, method='WLS', by_lines=False):

    ties = pd.DataFrame()
    fix_station = anchor

    if by_lines:
        groupby = raw_data.groupby(['instrument_serial_number', 'created', 'survey_name', 'operator', 'meter_type', 'line'])
    else:
        groupby = raw_data.groupby(['instrument_serial_number', 'created', 'survey_name', 'operator', 'meter_type'])
    
    for meter_created_survey_operator_meter_type, grouped in groupby:

        indices = grouped.index

        if anchor is None:
            fix_station = grouped.station.iloc[0]

        if by_lines:
            meter, created, survey, operator, meter_type, line = meter_created_survey_operator_meter_type
        else:
            meter, created, survey, operator, meter_type = meter_created_survey_operator_meter_type

        fitgrav, raw_data.loc[indices, 'resid'] = free_grav_fit(
            stations=grouped.station,
            gravity=grouped.corr_grav,
            date_time=grouped.date_time.apply(to_days),
            fix_station=fix_station,
            std=grouped.std_err,
            max_degree=2,
            method=method,
        )
        fitgrav['instrument_serial_number'] = meter
        fitgrav['survey_name'] = survey
        fitgrav['operator'] = operator
        fitgrav['meter_type'] = meter_type
        fitgrav['date_time'] = created.date()
        instr_height_from = grouped.loc[grouped.station == fix_station, 'instr_height'].mean()
        lat_from = grouped.loc[grouped.station == fix_station, 'lat'].mean()
        lon_from = grouped.loc[grouped.station == fix_station, 'lon'].mean()
        fitgrav['lat_from'] = lat_from
        fitgrav['lon_from'] = lon_from
        fitgrav['instr_height_from'] = instr_height_from
        for idx, row in fitgrav.iterrows():
            lat_to = grouped.loc[grouped.station == row.station_to, 'lat'].mean()
            lon_to = grouped.loc[grouped.station == row.station_to, 'lon'].mean()
            instr_height_to = grouped.loc[grouped.station == row.station_to, 'instr_height'].mean()
            fitgrav.loc[idx, 'lat_to'] = lat_to
            fitgrav.loc[idx, 'lon_to'] = lon_to
            fitgrav.loc[idx, 'instr_height_to'] = instr_height_to
            if by_lines:
                fitgrav.loc[idx, 'line'] = int(line)

        ties = pd.concat([ties, fitgrav], ignore_index=True)
    
    return ties
 