from datetime import timedelta as td
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString



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
        'residuals'
    )

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
