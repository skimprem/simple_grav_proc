from datetime import timedelta as td
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np


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

def vg_plot(coeffs, ties):

    figs = []
    for _, row in coeffs.iterrows():
        df = ties[(ties.meter == row.meter) & (ties.survey == row.survey)]
        y = np.linspace(0, 1.5, 50)
        b, a = row.b, row.a
        p = np.poly1d([b, a, 0])
        resid = row.resid.reshape((len(df), 2))
        # h_min = df[['from_height', 'to_height']].min().min() * 1e-3
        h_ref = 1
        # substruct = (p(h_min) - p(h_ref)) / (h_min - h_ref)
        substruct = p(h_ref)
        gp = lambda x: p(x) - x * substruct
        ua, ub = row.ua, row.ub
        cov = row.covab
        u = abs(h_ref - y) * np.sqrt(ub**2 + (y + h_ref)**2 * ua**2 + 2 * (h_ref + y) * cov)
        x = gp(y)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y)
        ax.fill_betweenx(y, x - u, x + u, alpha=0.2)
        for height_from, height_to, resid in zip(df.from_height, df.to_height, resid):
            heights = np.array([height_from, height_to]) * 1e-3
            ax.plot(gp(heights) + resid, heights, '.-')
        plt.title(f'Meter: {row.meter}, survey: {row.survey} (substract {substruct:.1f} $\mu$Gal/m)')
        plt.xlabel(f'Gravity, $\mu$Gal')
        plt.ylabel('Height, m')
        ax.set(xlim=(-10, 10), ylim=(0, 1.5))
        figs.append((fig, '_'.join([str(row.meter), str(row.survey)])))
        # fig.savefig('_'.join([str(row.meter), str(row.survey)])+'.png')
    return figs
    
