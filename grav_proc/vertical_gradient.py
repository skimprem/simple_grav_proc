import numpy as np
import pandas as pd
import statsmodels.api as sm

def get_vg(readings, method='WLS', max_degree=2, vg_max_degree=2):

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
        'data_file': [],
        'created_date': [],
        'operator': [],
   }

    group_by_meter_and_survey = readings.groupby(['instrument_serial_number', 'survey_name'])
    for meter_survey, grouped_by_meter_and_survey in group_by_meter_and_survey:
        meter, survey = meter_survey
        group_by_line = grouped_by_meter_and_survey.groupby('line')
        for line, grouped_by_line in group_by_line:
            grav = np.vstack(grouped_by_line.corr_grav)
            date_time = np.array(grouped_by_line.date_time.apply(lambda x: x.timestamp()/86400))
            drift_design = np.vander(date_time, max_degree + 1)
            change_stations = grouped_by_line.station.unique()
            change_heights = grouped_by_line.instr_height.unique()
            fix_station = change_stations[0]
            fix_height = change_heights[0]
            change_stations = change_stations[change_stations != fix_station]
            change_heights = change_heights[change_heights != fix_height]
            rows = []
            stations = grouped_by_line.station
            data_file = grouped_by_line.iloc[0].data_file
            created_date = grouped_by_line.iloc[0].created
            operator = grouped_by_line.iloc[0].operator
            for station in stations:
                row = []
                for change_station in change_stations:
                    if station == change_station:
                        row.append(1)
                    else:
                        row.append(0)
                rows.append(row)
            grav_design = np.array(rows)
            design = np.concatenate((grav_design, drift_design), axis=1)
            match method:
                case 'WLS':
                    model = sm.WLS(grav, design, weights=grouped_by_line.std_err**-2)
                case 'OLS':
                    model = sm.OLS(grav, design)
                case 'RLM':
                    model = sm.RLM(grav, design)
            result = model.fit()
            const = result.params[-1]
            std_const = result.bse[-1]
            drift = tuple(result.params[-(max_degree+1):-1])
            std_drift = tuple(result.bse[-(max_degree+1):-1])
            gravity = result.params[:-(max_degree+1)]
            std_gravity = result.bse[:-(max_degree+1)]
            stations_number = len(change_stations)
            for index, station, height in zip(range(stations_number), change_stations, change_heights):
                ties_dict['meter'].append(meter)
                ties_dict['survey'].append(survey)
                ties_dict['line'].append(line)
                ties_dict['from_point'].append(fix_station)
                ties_dict['to_point'].append(station)
                ties_dict['from_height'].append(fix_height)
                ties_dict['to_height'].append(height)
                ties_dict['gravity'].append(gravity[index])
                ties_dict['std_gravity'].append(std_gravity[index])
                ties_dict['drift'].append(drift)
                ties_dict['std_drift'].append(std_drift)
                ties_dict['const'].append(const)
                ties_dict['std_const'].append(std_const)
                ties_dict['data_file'].append(data_file)
                ties_dict['created_date'].append(created_date)
                ties_dict['operator'].append(operator)
    
    ties = pd.DataFrame(ties_dict)

    vg = pd.DataFrame()

    group_by_survey = ties.groupby('survey')
    for survey, grouped_by_survey in group_by_survey:
        from_height = np.vstack(grouped_by_survey.from_height * 1e-3)
        to_height = np.vstack(grouped_by_survey.to_height * 1e-3)
        heights = np.hstack((from_height, to_height)).flatten()
        to_gravity = np.vstack(grouped_by_survey.gravity)
        from_gravity = np.zeros_like(to_gravity)
        gravity = np.hstack((from_gravity, to_gravity)).flatten()
        coef_design = np.vander(heights, vg_max_degree + 1)[:,:-1]
        grouped_by_survey['line_meter'] = grouped_by_survey.apply(lambda x: '{line}_{meter}'.format(line=x.line, meter=x.meter), axis=1)
        grav_design = np.repeat(np.matrix(pd.get_dummies(grouped_by_survey.line_meter).astype(float)), 2, axis=0)
        design = np.concatenate((coef_design, grav_design), axis=1)
        # model = sm.OLS(gravity, design)
        model = sm.RLM(gravity, design)
        result = model.fit() 
        coefs = list(result.params[:vg_max_degree])[::-1]
        std_coefs = list(result.bse[:vg_max_degree])[::-1]
        cov_coefs = result.cov_params()[0][1]
        coefs_number = len(coefs)
        coef_names = list(map(chr, range(97, 97+coefs_number)))
        std_coef_names = ['u'+x for x in coef_names]
        columns = ['survey']+coef_names+std_coef_names+['covab', 'resid']
        resid = result.resid
        vg = pd.concat([vg, pd.DataFrame([[survey]+coefs+std_coefs+[cov_coefs]+[resid]], columns=columns)], axis=0)

    return ties, vg

def get_vg_by_meter(readings, method = 'WLS', max_degree=2, vg_max_degree=2):

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
        'data_file': [],
        'created_date': [],
        'operator': [],
   }

    group_by_meter_and_survey = readings.groupby(['instrument_serial_number', 'survey_name'])
    for meter_survey, grouped_by_meter_and_survey in group_by_meter_and_survey:
        meter, survey = meter_survey
        group_by_line = grouped_by_meter_and_survey.groupby('line')
        for line, grouped_by_line in group_by_line:
            grav = np.vstack(grouped_by_line.corr_grav)
            date_time = np.array(grouped_by_line.date_time.apply(lambda x: x.timestamp()/86400))
            drift_design = np.vander(date_time, max_degree + 1)
            change_stations = grouped_by_line.station.unique()
            change_heights = grouped_by_line.instr_height.unique()
            fix_station = change_stations[0]
            fix_height = change_heights[0]
            change_stations = change_stations[change_stations != fix_station]
            change_heights = change_heights[change_heights != fix_height]
            rows = []
            stations = grouped_by_line.station
            data_file = grouped_by_line.iloc[0].data_file
            created_date = grouped_by_line.iloc[0].created
            operator = grouped_by_line.iloc[0].operator
            for station in stations:
                row = []
                for change_station in change_stations:
                    if station == change_station:
                        row.append(1)
                    else:
                        row.append(0)
                rows.append(row)
            grav_design = np.array(rows)
            design = np.concatenate((grav_design, drift_design), axis=1)
            match method:
                case 'WLS':
                    model = sm.WLS(grav, design, weights=grouped_by_line.std_err**-2)
                case 'OLS':
                    model = sm.OLS(grav, design)
                case 'RLM':
                    model = sm.RLM(grav, design)
            result = model.fit()
            const = result.params[-1]
            std_const = result.bse[-1]
            drift = tuple(result.params[-(max_degree+1):-1])
            std_drift = tuple(result.bse[-(max_degree+1):-1])
            gravity = result.params[:-(max_degree+1)]
            std_gravity = result.bse[:-(max_degree+1)]
            stations_number = len(change_stations)
            for index, station, height in zip(range(stations_number), change_stations, change_heights):
                ties_dict['meter'].append(meter)
                ties_dict['survey'].append(survey)
                ties_dict['line'].append(line)
                ties_dict['from_point'].append(fix_station)
                ties_dict['to_point'].append(station)
                ties_dict['from_height'].append(fix_height)
                ties_dict['to_height'].append(height)
                ties_dict['gravity'].append(gravity[index])
                ties_dict['std_gravity'].append(std_gravity[index])
                ties_dict['drift'].append(drift)
                ties_dict['std_drift'].append(std_drift)
                ties_dict['const'].append(const)
                ties_dict['std_const'].append(std_const)
                ties_dict['data_file'].append(data_file)
                ties_dict['created_date'].append(created_date)
                ties_dict['operator'].append(operator)
    
    ties = pd.DataFrame(ties_dict)

    vg = pd.DataFrame()

    group_by_meter_and_survey = ties.groupby(['meter', 'survey'])
    for meter_survey, grouped_by_meter in group_by_meter_and_survey:
        meter, survey = meter_survey
        from_height = np.vstack(grouped_by_meter.from_height * 1e-3)
        to_height = np.vstack(grouped_by_meter.to_height * 1e-3)
        heights = np.hstack((from_height, to_height)).flatten()
        to_gravity = np.vstack(grouped_by_meter.gravity)
        from_gravity = np.zeros_like(to_gravity)
        gravity = np.hstack((from_gravity, to_gravity)).flatten()
        coef_design = np.vander(heights, vg_max_degree + 1)[:,:-1]
        grav_design = np.repeat(np.matrix(pd.get_dummies(grouped_by_meter.line).astype(float)), 2, axis=0)
        design = np.concatenate((coef_design, grav_design), axis=1)
        # model = sm.OLS(gravity, design)
        model = sm.RLM(gravity, design)
        result = model.fit() 
        coefs = list(result.params[:vg_max_degree])[::-1]
        std_coefs = list(result.bse[:vg_max_degree])[::-1]
        cov_coefs = result.cov_params()[0][1]
        coefs_number = len(coefs)
        coef_names = list(map(chr, range(97, 97+coefs_number)))
        std_coef_names = ['u'+x for x in coef_names]
        columns = ['meter', 'survey']+coef_names+std_coef_names+['covab', 'resid']
        resid = result.resid
        vg = pd.concat([vg, pd.DataFrame([[meter, survey]+coefs+std_coefs+[cov_coefs]+[resid]], columns=columns)], axis=0)

    return ties, vg

