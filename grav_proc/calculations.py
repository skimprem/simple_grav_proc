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

def gravfit(input_stations, input_grav, input_std, time_, max_degree=2):

    stations = np.array(input_stations)
    unique_stations = input_stations.unique()
    first_station = unique_stations[0]
    defined_stations = unique_stations[unique_stations != first_station].T
    stations_number = stations.size
    defined_stations_number = defined_stations.size

    rows = []
    for station in stations:
        row = []
        for defined_station in defined_stations:
            if station == defined_station:
                row.append(1)
            else:
                row.append(0)
        rows.append(row)
    observation_matrix = np.array(rows)

    time_matrix = np.vstack(time_ - time_.iloc[0])
    if max_degree > 1:
        for degree in range(2, max_degree + 1):
            time_matrix = np.hstack((time_matrix, np.power(time_matrix, degree)))
    
    ones = np.ones(shape=(stations_number, 1))
    design_matrix = np.concatenate((time_matrix), axis=1)

    model = sm.OLS(input_grav, design_matrix)

    result = model.fit()

    ties = {
        'from_station': [],
        'to_station': [],
        'grav': [],
        'std_err': []
    }

    for index, station_name in enumerate(defined_stations):
        ties['from_station'].append(first_station)
        ties['to_station'].append(station_name)
        ties['grav'].append(result.params[index])
        ties['std_err'].append(result.bse[index])
    
    return pd.DataFrame(ties)

def get_vg(readings, max_degree=2, vg_max_degree=2):

    ties_dict = {
        'meter': [],
        'survey': [],
        'line': [],
        'from_point': [],
        'to_point': [],
        'from_height': [],
        'to_height': [],
        'gravity': [],
        'std_gravity': [],
        'drift': [],
        'std_drift': [],
        'const': [],
        'std_const': [],
   }

    # group_by_meter_and_survey = readings.groupby(['instrument_serial_number', 'survey_name'])
    # for meter_survey, grouped_by_meter_and_survey in group_by_meter_and_survey:
    #     meter, survey = meter_survey
    #     group_by_line = grouped_by_meter_and_survey.groupby('line')
    #     for line, grouped_by_line in group_by_line:
    #         grav = np.vstack(grouped_by_line.corr_grav)
    #         date_time = np.array(grouped_by_line.date_time.apply(lambda x: x.timestamp()/86400))
    #         drift_design = np.vander(date_time, max_degree + 1)
    #         change_stations = grouped_by_line.station.unique()
    #         change_heights = grouped_by_line.instr_height.unique()
    #         fix_station = change_stations[0]
    #         fix_height = change_heights[0]
    #         change_stations = change_stations[change_stations != fix_station]
    #         change_heights = change_heights[change_heights != fix_height]
    #         rows = []
    #         stations = grouped_by_line.station
    #         for station in stations:
    #             row = []
    #             for change_station in change_stations:
    #                 if station == change_station:
    #                     row.append(1)
    #                 else:
    #                     row.append(0)
    #             rows.append(row)
    #         grav_design = np.array(rows)
    #         design = np.concatenate((grav_design, drift_design), axis=1)
    #         model = sm.OLS(grav, design)
    #         result = model.fit()
    #         const = result.params[-1]
    #         std_const = result.bse[-1]
    #         drift = tuple(result.params[-(max_degree+1):-1])
    #         std_drift = tuple(result.bse[-(max_degree+1):-1])
    #         gravity = result.params[:-(max_degree+1)]
    #         std_gravity = result.bse[:-(max_degree+1)]
    #         stations_number = len(change_stations)
    #         for index, station, height in zip(range(stations_number), change_stations, change_heights):
    #             ties_dict['meter'].append(meter)
    #             ties_dict['survey'].append(survey)
    #             ties_dict['line'].append(line)
    #             ties_dict['from_point'].append(fix_station)
    #             ties_dict['to_point'].append(station)
    #             ties_dict['from_height'].append(fix_height)
    #             ties_dict['to_height'].append(height)
    #             ties_dict['gravity'].append(gravity[index])
    #             ties_dict['std_gravity'].append(std_gravity[index])
    #             ties_dict['drift'].append(drift)
    #             ties_dict['std_drift'].append(std_drift)
    #             ties_dict['const'].append(const)
    #             ties_dict['std_const'].append(std_const)
    
    ties_dict['meter'].append('001')
    ties_dict['survey'].append('001')
    ties_dict['line'].append('001')
    ties_dict['from_point'].append('1')
    ties_dict['to_point'].append('2')
    ties_dict['from_height'].append(270.0)
    ties_dict['to_height'].append(719.0)
    ties_dict['gravity'].append(-144.2)
    ties_dict['std_gravity'].append(0.2)
    ties_dict['drift'].append(144.2)
    ties_dict['std_drift'].append(0.2)
    ties_dict['const'].append(0.0)
    ties_dict['std_const'].append(0.0)

    ties_dict['meter'].append('001')
    ties_dict['survey'].append('001')
    ties_dict['line'].append('001')
    ties_dict['from_point'].append('2')
    ties_dict['to_point'].append('3')
    ties_dict['from_height'].append(719.0)
    ties_dict['to_height'].append(1300.0)
    ties_dict['gravity'].append(-189.2)
    ties_dict['std_gravity'].append(0.2)
    ties_dict['drift'].append(144.2)
    ties_dict['std_drift'].append(0.2)
    ties_dict['const'].append(0.0)
    ties_dict['std_const'].append(0.0)

    ties_dict['meter'].append('001')
    ties_dict['survey'].append('001')
    ties_dict['line'].append('001')
    ties_dict['from_point'].append('1')
    ties_dict['to_point'].append('3')
    ties_dict['from_height'].append(270.0)
    ties_dict['to_height'].append(1300.0)
    ties_dict['gravity'].append(-333.8)
    ties_dict['std_gravity'].append(0.2)
    ties_dict['drift'].append(144.2)
    ties_dict['std_drift'].append(0.2)
    ties_dict['const'].append(0.0)
    ties_dict['std_const'].append(0.0)

    print(ties_dict)

    ties = pd.DataFrame(ties_dict)
    print(ties)

    vg_dict = {
        'meter': [],
        'survey': [],
        'coefs': [],
        'std_coefs': [],
        'cov_coefs': [],
    }
    group_by_meter_and_survey = ties.groupby(['meter', 'survey'])
    for meter_survey, grouped_by_meter in group_by_meter_and_survey:
        meter, survey = meter_survey
        grav = grouped_by_meter.gravity
        from_height = np.vstack(grouped_by_meter.from_height * 1e-3)
        to_height = np.vstack(grouped_by_meter.to_height * 1e-3)

        # points = np.concatenate((np.vstack(grouped_by_meter.from_point), np.vstack(grouped_by_meter.to_point)), axis=0)
        # rows = []
        # for from_point, to_point in zip(grouped_by_meter.from_point, grouped_by_meter.to_point):
        #     row = []
        #     for point in points:
        #         if from_point == point:
        #             row.append(-1)
        #         elif to_point == point:
        #             row.append(1)
        #         else:
        #             row.append(0)
        #     rows.append(row)
        
        # grav_design = np.array(rows)
        # print(grav_design)
        
        coef_design = to_height - from_height
        if vg_max_degree > 1:
            for degree in range(2, vg_max_degree + 1):
                coef_design = np.hstack((coef_design, to_height**degree - from_height**degree))
        ones = np.ones(shape=(grav.size, 1))
        design = np.concatenate((coef_design, ones), axis=1)
        # design = np.concatenate((grav_design, coef_design, ones), axis=1)
        model = sm.OLS(grav, design)
        result = model.fit() 
        print(result.params)
        coefs = result.params
        std_coefs = result.bse
        vg_dict['meter'].append(meter)
        vg_dict['survey'].append(survey)
        vg_dict['coefs'].append(list(coefs))
        vg_dict['std_coefs'].append(list(std_coefs))
        print(std_coefs)
        cov_coefs = result.cov_params().iloc[0].iloc[1]
        # rows = []
        # for index, row in cov_params.iterrows():
        #     rows.append([x for x in row])
        vg_dict['cov_coefs'].append(cov_coefs)

    vg = pd.DataFrame(vg_dict)
    return ties, vg


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