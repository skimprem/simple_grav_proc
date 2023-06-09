import pandas as pd
import re
from datetime import datetime
import os


def cg6_data_file_reader(path_to_input_data_file):
    # os.path.abspath("insert-file-name-here")
    input_data_file = open(os.path.abspath(path_to_input_data_file), 'r')
    columns = ['survey_name', 'instrument_serial_number', 'created', 'operator', 'g_cal1', 'g_off', 'g_ref', 'x_scale', 'y_scale', 'x_offset', 'y_offset', 'temperature_coefficient',
               'temperature_scale', 'drift_rate', 'drift_zero_time', 'firmware_version', 'station', 'date_time', 'corr_grav', 'line', 'std_dev', 'std_err', 'raw_grav', 'x', 'y',
               'sensor_temp', 'tide_corr', 'tilt_corr', 'temp_corr', 'drift_corr', 'measur_dur', 'instr_height', 'lat_user', 'lon_user', 'elev_user', 'lat_gps', 'lon_gps', 'elev_gps', 'corrections', 'data_file']
    cg6_data = pd.DataFrame(columns=columns)
    count = 0
    for line in input_data_file:
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
                    continue
                case 'Instrument Serial Number':
                    instrument_serial_number = int(split_line[1])
                    continue
                case 'Created':
                    created = datetime.strptime(
                        split_line[1], "%Y-%m-%d %H:%M:%S")
                    continue
                case 'Operator':
                    operator = split_line[1]
                    continue
                case 'Gcal1 [mGal]':
                    gcal1 = float(split_line[1])
                    continue
                case 'Goff [ADU]':
                    goff = float(split_line[1])
                    continue
                case 'Gref [mGal]':
                    gref = float(split_line[1])
                    continue
                case 'X Scale [arc-sec/ADU]':
                    x_scale = float(split_line[1])
                    continue
                case 'Y Scale [arc-sec/ADU]':
                    y_scale = float(split_line[1])
                    continue
                case 'X Offset [ADU]':
                    x_offset = float(split_line[1])
                    continue
                case 'Y Offset [ADU]':
                    y_offset = float(split_line[1])
                    continue
                case 'Temperature Coefficient [mGal/mK]':
                    temperature_coefficient = float(split_line[1])
                    continue
                case 'Temperature Scale [mK/ADU]':
                    temperature_scale = float(split_line[1])
                    continue
                case 'Drift Rate [mGal/day]':
                    drift_rate = float(split_line[1])
                    continue
                case 'Drift Zero Time':
                    drift_zero_time = datetime.strptime(
                        split_line[1], "%Y-%m-%d %H:%M:%S")
                    continue
                case 'Firmware Version':
                    firmware_version = split_line[1]
                    continue
        else:
            if not line.strip():
                continue
            split_line = line.split()
            try:
                station, date_, time_, corrgrav, line_, stddev, stderr, rawgrav, x, y, sensortemp, tidecorr, tiltcorr, tempcorr, driftcorr, measurdur, instrheight, latuser, \
                lonuser, elevuser, latgps, longps, elevgps, corrections = split_line
            except ValueError:
                print(f'Warning: ValueError at line {count}')
                continue

            date_time = datetime.strptime(
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
            cg6_data.loc[len(cg6_data)] = [survey_name, instrument_serial_number, created, operator, gcal1, goff, gref, x_scale, y_scale,
                                            x_offset, y_offset, temperature_coefficient, temperature_scale, drift_rate, drift_zero_time, firmware_version, station, date_time, corrgrav,
                                            line_, stddev, stderr, rawgrav, x, y, sensortemp, tidecorr, tiltcorr, tempcorr, driftcorr, measurdur, instrheight, latuser, lonuser, elevuser, latgps,
                                            longps, elevgps, corrections, path_to_input_data_file]
    input_data_file.close()
    cg6_data.set_index('date_time', inplace=True)

    return cg6_data

def get_readings(cg6_data):
    readings = pd.DataFrame(columns=['created', 'survey_name', 'operator', 'instrument_serial_number',
                            'instr_height', 'line', 'station', 'date_time', 'corr_grav', 'data_file'])
    reading_count = 0
    for line in cg6_data.line.unique():
        line_data = cg6_data[cg6_data.line == line]
        trigger = False
        count = 0
        for index, row in line_data.iterrows():
            if index == line_data.index[-1]:
                station_mean = line_data.corr_grav.loc[first_index:index].mean(
                )
                height_mean = line_data.instr_height.loc[first_index:index].mean(
                )
                mean_time = first_index + (index - first_index) / 2
                readings.loc[reading_count] = [datetime.date(
                    row.created), row.survey_name, row.operator, row.instrument_serial_number, height_mean, row.line, row.station, mean_time, station_mean, row.data_file]
                reading_count += 1
                break
            if row.station == line_data.station.iloc[count + 1]:
                count += 1
                if not trigger:
                    trigger = True
                    first_index = index
            else:
                trigger = False
                station_mean = line_data.corr_grav.loc[first_index:index].mean(
                )
                height_mean = line_data.instr_height.loc[first_index:index].mean(
                )
                mean_time = first_index + (index - first_index) / 2
                readings.loc[reading_count] = [datetime.date(
                    row.created), row.survey_name, row.operator, row.instrument_serial_number, height_mean, row.line, row.station, mean_time, station_mean, row.data_file]
                reading_count += 1
                count += 1
    return readings


def get_ties(readings):
    ties = pd.DataFrame(columns=['created', 'survey_name', 'operator', 'instrument_serial_number',
                        'instr_height_from', 'instr_height_to', 'line', 'station_from', 'station_to', 'tie', 'data_file'])
    tie_count = 0
    for line in readings.line.unique():
        line_readings = readings[readings.line == line]
        loops = []
        for index, row in line_readings.iterrows():
            if len(loops) == 0:
                loops.append({'date': row['created'], 'survey_name': row.survey_name, 'operator': row.operator, 'station': row.station, 'index': index, 'date_time': row.date_time,
                             'corr_grav': row.corr_grav, 'instr_height': row.instr_height, 'instrument_serial_number': row.instrument_serial_number, 'data_file': row.data_file})
                continue
            if row.station == loops[0]['station']:
                begin_index = loops[0]['index']
                factor = (row.corr_grav - line_readings.corr_grav.loc[begin_index]) / (datetime.timestamp(
                    row.date_time) - datetime.timestamp(line_readings.date_time.loc[begin_index]))
                for reading in loops[1:]:
                    correction = factor * (datetime.timestamp(reading['date_time']) - datetime.timestamp(
                        line_readings.date_time.loc[begin_index]))
                    tie = reading['corr_grav'] - \
                        loops[0]['corr_grav'] + correction
                    level_from = loops[0]['instr_height']
                    level_to = reading['instr_height']
                    date = reading['date']
                    site = reading['survey_name']
                    operator = reading['operator']
                    meter = reading['instrument_serial_number']
                    data_file = reading['data_file']
                    ties.loc[tie_count] = [date, site, operator, meter, level_to,
                                           level_from, row.line, row.station, reading['station'], tie, data_file]
                    tie_count += 1
                loops.pop(0)
                loops.append({'date': row['created'], 'survey_name': row.survey_name, 'operator': row.operator, 'station': row.station, 'index': index, 'date_time': row.date_time,
                             'corr_grav': row.corr_grav, 'instr_height': row.instr_height, 'instrument_serial_number': row.instrument_serial_number, 'data_file': row.data_file})
            else:
                loops.append({'date': row['created'], 'survey_name': row.survey_name, 'operator': row.operator, 'station': row.station, 'index': index, 'date_time': row.date_time,
                             'corr_grav': row.corr_grav, 'instr_height': row.instr_height, 'instrument_serial_number': row.instrument_serial_number, 'data_file': row.data_file})
    return ties


def get_mean_ties(ties):
    
    for index, row in ties.iterrows():
        from_station = row['station_from']
        to_station = row['station_to']
        for tie_index, tie_row in ties.iterrows():
            if tie_row['station_from'] == to_station and tie_row['station_to'] == from_station:
                ties.loc[tie_index] = [tie_row['created'], tie_row['survey_name'], tie_row['operator'], tie_row['instrument_serial_number'],
                                       row['instr_height_to'], row['instr_height_from'], tie_row['line'], from_station, to_station, -tie_row['tie'], tie_row['data_file']]
    result = pd.DataFrame(columns=['station_from', 'station_to', 'created', 'station', 'operator',
                          'instrument_serial_number', 'line', 'instr_height_from', 'instr_height_to', 'tie', 'std', 'source'])

    means = []
    count = 1
    for line in ties.line.unique():
        line_ties = ties[ties.line == int(line)]
        group_mean = line_ties.groupby(
            ['station_from', 'station_to'], as_index=False)
        mean = group_mean.agg({'created': 'last', 'survey_name': 'last', 'operator': 'last', 'instrument_serial_number': 'last',
                              'line': 'last', 'instr_height_from': 'mean', 'instr_height_to': 'mean', 'tie': ['mean', 'std'], 'data_file': 'last'})
        count += 1

        mean.columns = ['station_from', 'station_to', 'created', 'survey_name', 'operator',
                        'instrument_serial_number', 'line', 'instr_height_from', 'instr_height_to', 'tie', 'std', 'data_file']
        means.append(mean)
    result = pd.concat(means, ignore_index=True)
    return result


def get_ties_sum(ties):

    return sort_ties(ties).tie.sum()

def sort_ties(ties):
    sort_ties = ties
    index = 0
    tie = sort_ties.iloc[index]
    previous_to = tie.station_to
    index += 1
    while index < len(sort_ties):
        tie = sort_ties.iloc[index]
        if previous_to == tie.station_from:
            previous_to = tie.station_to
            index += 1
        elif previous_to == tie.station_to:
            sort_ties.iloc[index] = reverse_tie(tie)
            previous_to = tie.station_from
            index += 1
        else:
            last_ties = sort_ties.iloc[index:].shift(-1, fill_value=9999)
            last_ties.iloc[-1] = tie
            sort_ties = pd.concat([sort_ties.iloc[:index], last_ties])
    return sort_ties

def reverse_tie(tie):
    reverse_tie = [
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
        tie.data_file
    ]
    return reverse_tie

def print_means(means):

    columns = means.columns[:-1]
    headers = ['From', 'To', 'Date', 'Survey', 'Operator', 'S/N', 'Line',
               'Height From (mm)', 'Height To (mm)', 'Tie (uGals)', 'SDev (uGals)']
    print(means[columns].to_markdown(index=False,
          headers=headers, tablefmt="simple", floatfmt='.2f'))
    return


def make_output(means, filename):
    columns = means.columns[:-1]
                # ['station_from', 'station_to', 'created', 'survey_name', 'operator',
            #    'instrument_serial_number', 'line', 'instr_height_from', 'instr_height_to', 'tie', 'std']
    headers = ['From', 'To', 'Date', 'Survey', 'Operator', 'S/N', 'Line',
               'Height From (mm)', 'Height To (mm)', 'Tie (uGals)', 'SDev (uGals)']
    means[columns].to_markdown(
        filename, index=False, headers=headers, tablefmt="simple", floatfmt=".1f")
    return


def make_vgfit_input(means, filename):
    means_to_vgfit = means
    means_to_vgfit.columns = ['from', 'to', 'date', 'station', 'observer',
                     'gravimeter', 'runn', 'level_1', 'level_2', 'delta_g', 'std', 'source']
    means_to_vgfit[means_to_vgfit.columns[2:]].to_csv(filename, index=False)
    return means_to_vgfit
