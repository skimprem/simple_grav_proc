import pandas as pd
from datetime import datetime as dt
# from time import time as tm
import numpy as np
import networkx as nx
import statsmodels.api as sm


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
                    'StdErr',
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
                'std_err',
                'data_file',
                'lat_user',
                'lon_user',
                'meter_type'
            ]

            data.columns = headers
            for index, row in data.iterrows():
                data.loc[index, 'corr_grav'] = row.corr_grav * 1e3
                data.loc[index, 'std_err'] = row.std_err * 1e3
                data.loc[index, 'instr_height'] = row.instr_height * 1e3
            return data
        case 'CG5':
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
                    'StdErr',
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
                'std_err',
                'data_file',
                'lat_user',
                'lon_user',
                'meter_type'
            ]

            data.loc[:, 'CorrGrav'] = data.loc[:, 'CorrGrav'] * 1000.0
            data.loc[:, 'StdErr']  = data.loc[:, 'StdErr'] * 1000.0
            data.loc[:, 'InstrHeight'] = data.loc[:, 'InstrHeight'] * 1000.0

            data.columns = headers

            return data



def get_meters_readings(cg_data):
    ''' Get mean values of signals of readings from different meters '''

    readings = pd.DataFrame()
    # for meter in cg_data.instrument_serial_number.unique():
    #     meter_data = cg_data[cg_data.instrument_serial_number == meter]
    #     meter_readings = get_readings(meter_data)
    #     readings = pd.concat([readings, meter_readings])

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
    
    # for meter in readings.instrument_serial_number.unique():
    #     meter_readings = readings[readings.instrument_serial_number == meter]
    #     meter_ties = get_ties(meter_readings)
    #     ties = pd.concat([ties, meter_ties])
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
    for line in ties.line.unique():
        line_ties = ties[ties.line == int(line)]
        # print(line_ties.date_to)
        # print((line_ties.loc[0, 'date_to']))
        # for index_line_tie, _ in line_ties.iterrows():
        #     line_ties.loc[index_line_tie, 'date_to'] =\
        #         dt.date(line_ties.loc[index_line_tie, 'date_to'])
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


def gravfit(grav, date_time):

    x = np.array(grav)

    print(x)
    a = np.append(np.array(date_time), np.ones((np.size(grav), 1)), axis=1)
    print(a)

    model = sm.RLM(x, a)

    res = model.fit()

    return res
