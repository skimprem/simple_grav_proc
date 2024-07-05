from datetime import timedelta as td
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import pandas as pd
import contextily as cx
from cartopy import crs as ccrs
import cartopy.io.img_tiles as cimgt
# from cartopy.io.img_tiles import GoogleTiles
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

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



def get_map(ties):
    ''' Get map of ties scheme '''

    stations_from = ties[['station_from', 'lat_from', 'lon_from']]
    stations_from.columns = ['station', 'lat', 'lon']
    stations_to = ties[['station_to', 'lat_to', 'lon_to']]
    stations_to.columns = ['station', 'lat', 'lon']
    stations = pd.concat([stations_from, stations_to], ignore_index=True).groupby('station').mean()

    lines = ties[['station_from', 'station_to']].drop_duplicates(ignore_index=True)
    
    fig = plt.figure(figsize=(15, 15))
    xmin, xmax, ymin, ymax = stations.lon.min(), stations.lon.max(), stations.lat.min(), stations.lat.max()
    dx = xmax - xmin
    dy = ymax - ymin
    if dx < 2 * dy:
        offsety = dy * 0.1
        ymin, ymax = ymin - offsety, ymax + offsety
        dy = ymax - ymin
        dx = dy * 16 / 9
        centerx = xmin + (xmax - xmin) / 2
        xmin, xmax = centerx - dx / 2, centerx + dx / 2
    else:
        offsetx = dx * 0.1
        xmin, xmax = xmin - offsetx, xmax + offsetx
        dx = xmax - xmin
        dy = dx * 9 / 16
        centery = ymin + (ymax - ymin) / 2
        ymin, ymax = centery - dy / 2, centery + dy / 2
       
    extent = [xmin, xmax, ymin, ymax]
    request = cimgt.OSM()
    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)

    if dx < 0.2:
        zoom = 13
    elif dx < 0.3 and dx > 0.2:
        zoom = 12
    elif dx < 0.7 and dx > 0.3:
        zoom = 11
    elif dx < 1 and dx > 0.7:
        zoom = 10
    else:
        zoom = 8

    ax.add_image(request, zoom)

    for _, row in lines.iterrows():
        x_from = stations.loc[row.station_from, 'lon']
        y_from = stations.loc[row.station_from, 'lat']
        x_to = stations.loc[row.station_to, 'lon']
        y_to = stations.loc[row.station_to, 'lat']

        # tie = ties[
        #     (ties['station_from'] == row.station_from) & \
        #     (ties['station_to'] == row.station_to)
        # ][['tie']].mean()
        ax.plot([x_from, x_to], [y_from, y_to], '-ok', mfc='w', transform=ccrs.PlateCarree())
   
    for idx, row in stations.iterrows():
        ax.annotate(idx, xy=(row.lon, row.lat),
                    xycoords='data', xytext=(1.5, 1.5),
                    textcoords='offset points',
                    fontsize=14,
                    color='k', transform=ccrs.PlateCarree())
    # plt.show()    
    return fig


def vg_plot(coeffs, ties, by_meter=False):

    figs = []
    for _, row in coeffs.iterrows():
        if by_meter:
            df = ties[(ties.meter == row.meter) & (ties.survey == row.survey)]
        else:
            df = ties[ties.survey == row.survey]
        y = np.linspace(0, 1.5, 50)
        b, a = row.b, row.a
        p = np.poly1d([b, a, 0])
        resid = row.resid.reshape((int(len(row.resid)/2), 2))
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
        if by_meter:
            plt.title(f'Meter: {row.meter}, survey: {row.survey} (substract {substruct:.1f} uGal/m)')
        else:
            plt.title(f'Survey: {row.survey} (substract {substruct:.1f} uGal/m)')
        plt.xlabel(f'Gravity [uGal]')
        plt.ylabel('Height [m]')
        low, high = plt.xlim()
        bound = max(abs(low), abs(high))
        ax.set(xlim=(-bound, bound), ylim=(0, 1.5))
        fig.tight_layout()
        figs.append((fig, row.survey))
        plt.close()
    return figs
    
def residuals_plot(raw_data):
    meters = raw_data['instrument_serial_number'].unique()
    meter_number = {}
    for index, meter in enumerate(meters):
        meter_number[meter] = index
    fig, ax = plt.subplots(nrows=len(meters), figsize=(16, 8), layout='constrained')
    fig.supylabel('Residuals [uGal]')
    fig.supxlabel('Date Time')

    for meter_created, grouped in raw_data.groupby(['instrument_serial_number', 'created']):
        meter, created = meter_created
        for station, grouped_by_station in grouped.groupby('station'):
            if len(meters) > 1:
                ax[meter_number[meter]].set_title(f'CG-6 #{meter}', loc='left')
                ax[meter_number[meter]].plot(grouped_by_station['date_time'], grouped_by_station['resid'], '.', label=station)
                # ax[meter_number[meter]].legend(loc='upper right')
            else:
                ax.set_title(f'CG-6 #{meter}', loc='left')
                ax.plot(grouped_by_station['date_time'], grouped_by_station['resid'], '.', label=station)
                # ax.legend(loc='upper right')
    fig.legend()
    fig.tight_layout()

    return fig
