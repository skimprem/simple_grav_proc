import numpy as np
import pandas as pd
from grav_proc.calculations import get_ties_sum


def get_report(means):
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
        'err'
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
        'SErr (uGals)'
    ]
    report = f'\nThe mean ties between the stations:\n==================================='
    means = means.replace(np.nan, None)
    means_table = means[columns].to_markdown(
        index=False,
        headers=headers,
        tablefmt="simple",
        floatfmt=".1f")
    report = f'{report}\n{means_table}'
    ties_sums = pd.DataFrame()

    group_by_meters = means.groupby('instrument_serial_number')

    for meter, meter_means in group_by_meters:
        meter_ties_sums = get_ties_sum(meter_means)
        if len(meter_ties_sums):
            ties_sums = pd.concat([ties_sums, meter_ties_sums])
    if len(ties_sums):
        report = f'{report}\n\nSum of the cicle ties:\n======================\n'
        headers = ['Meter', 'Cicles', 'Sum (uGals)']
        sums_table = ties_sums.to_markdown(
                index=False,
                headers=headers,
                tablefmt="simple",
                floatfmt=".2f")
        report = report + sums_table
    return report


def make_vgfit_input(means, filename):
    ''' Make CSV file for vg_fit utilite '''
    columns = [
        'created_date',
        'survey',
        'operator',
        'meter',
        'line',
        'from_height',
        'to_height',
        'gravity',
        'std_gravity',
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
        'err',
        'source'
    ]
    means_to_vgfit.to_csv(filename, index=False)
    return means_to_vgfit


def make_vg_ties_report(ties, output_file, verbose=False):
    columns = [
        'created_date',
        'survey',
        'operator',
        'meter',
        'line',
        'from_height',
        'to_height',
        'gravity',
        'std_gravity',
        'data_file'
    ]

    header = [
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

    ties.to_csv(
        output_file,
        columns=columns,
        header=header,
        index=False,
        date_format='%Y-%m-%d'
    )

    if verbose:
        print()
        print(ties[columns].to_markdown(index=False, tablefmt='simple', floatfmt='.3f'))
        print()

def make_vg_coeffs_report(coeffs, output_file, verbose=False):
    # coeffs[['meter', 'survey', 'a', 'b', 'ua', 'ub', 'covab']].to_csv(
    coeffs[['survey', 'a', 'b', 'ua', 'ub', 'covab']].to_csv(
        output_file,
        index=False,
        float_format='%.3f'
    )

    if verbose:
        print()
        # print(coeffs[['meter', 'survey', 'a', 'b', 'ua', 'ub', 'covab']].to_markdown(
        print(coeffs[['survey', 'a', 'b', 'ua', 'ub', 'covab']].to_markdown(
            index=False,
            tablefmt='simple',
            floatfmt='.1f'
        ))
        print()
