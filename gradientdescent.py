""" Execute gradient descent on earth geometry """

import sys
import csv
import rasterio
import argparse
import numpy as np
from pylab import plot, show, xlabel, ylabel
from scipy.optimize import minimize
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='output.csv')
parser.add_argument('--method', type=str, default='CG')
parser.add_argument('lat', type=float)
parser.add_argument('lon', type=float)
parser.add_argument('tif', type=str)
args = parser.parse_args()

src = rasterio.open(args.tif)
band = src.read(1)


def elevation(x):
    lat, lon = x
    vals = src.index(lon, lat)
    return band[vals]


def estimate_grad(x):
    lat, lon = x
    elev1 = elevation([lat + 0.001, lon])
    elev2 = elevation([lat - 0.001, lon])
    elev3 = elevation([lat, lon + 0.001])
    elev4 = elevation([lat, lon - 0.001])

    lat_slope = elev1 / elev2 - 1
    lon_slope = elev3 / elev4 - 1
    return np.array([lat_slope, lon_slope])


#  def minimize(fn, x0, jac=None, callback=None, **kwargs):


history = []
def record(x):
    global history
    lat, lon = x
    history += [{'elevation': elevation(x), 'lat': lat, 'lon': lon}]

location = np.array([args.lat, args.lon])
location = minimize(elevation, location, jac=estimate_grad, callback=record,
                    method=args.method)

with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for datum in history:
        if datum['lat'] != 0 and datum['lon'] != 0:
            writer.writerow([datum['lat'], datum['lon']])

df = pd.DataFrame(history)
df.to_csv('history_' + args.output)
