import scipy.io as sio
import numpy as np
import os
import pandas as pd

path_obs = r'./data/aus0079_timeseries_angourie'

# Carrega arquivos
files = os.listdir(path_obs)[4:-1]

obs = None

for i, f in enumerate(files):
    df = pd.read_csv(os.path.join(path_obs, f))
    df['Datetime'] = pd.to_datetime(df['dates UTC'])
    df = df.set_index('Datetime')
    df = df.drop(columns=['dates UTC'])

    df = df.rename(columns={' chainage (m)': f'transect_{i}'})

    # Aplicando o detrend individualmente
    mean_pos = df[f'transect_{i}'].mean()
    # df[f'transect_{i}'] -= mean_pos

    # Concatenar ao dataframe geral
    if obs is None:
        obs = df
    else:
        obs = pd.concat([obs, df], axis=1)

# Aqui obs já está detrend corretamente.
obs = obs.dropna(how='all').reset_index()

# Continuar com ondas (waves)
waves = sio.loadmat(r"./data/Angourie_ERA5_Plataforma_-29.5_153.5.mat")

time_w = waves['time'].flatten()
hs = waves['hs'].flatten()
ddir = waves['dir'].flatten()
tp = 1 / waves['fp'].flatten()

# Convertendo para datetime
time_w = pd.to_datetime(time_w - 719529, unit='d').round('s').to_pydatetime()

# Agora adicionar timestamp ao obs
obs['Timestamp'] = obs['Datetime'].apply(lambda x: x.timestamp())


from scipy.stats import circmean

obs_ = obs.drop(columns= ['Datetime', 'Timestamp']).mean(axis = 1).values

# vobs_ = vobs.drop(columns= ['Datetime', 'Timestamp']).mean(axis = 1).values

# hs = Hs['Transect5']
# obs_ = obs['Transect5']


time = time_w
time_obs = obs['Datetime'].values
# vtime_obs = vobs['Datetime'].values

time_obs = time_obs[~np.isnan(obs_)]
obs_ = obs_[~np.isnan(obs_)]

# vtime_obs = vtime_obs[~np.isnan(vobs_)]
# vobs_ = vobs_[~np.isnan(vobs_)]

from IHSetUtils import wMOORE

d50 = 0.3e-3
ws = wMOORE(d50)
omega = hs / (tp * ws)

# now we reduce the data (timestep == 1h) to timestep = 1d
# we have to get the mean for each day

time = pd.to_datetime(time)

time = pd.Series(time)
time = time.dt.floor('d')

hs = pd.Series(hs)
ddir = pd.Series(ddir)
tp = pd.Series(tp)
omega = pd.Series(omega)

hs = hs.groupby(time).mean()
ddir = ddir.groupby(time).mean()
tp = tp.groupby(time).mean()
omega = omega.groupby(time).mean()

hs = hs.values
ddir = ddir.values
tp = tp.values
omega = omega.values
time = time.groupby(time).mean()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 4))


ax.scatter(time_obs, obs_, s=1, marker='s', color='black', label='Observed')
# ax.scatter(vobs['Datetime'], vobs.transect_11, s=1, marker='s', color='red', label='Observed')

plt.show()

import geopandas as gpd
import numpy as np

path = r'./data/transects.geojson'

gdf = gpd.read_file(path)
# gdf

gdf = gdf[gdf['id'].str.startswith('aus0079')]
gdf = gdf.sort_values('id')

# now we hace to extract the initial and final coordinates of each transect and project them to UTM
import pyproj

def project_coords(coords):
    p = pyproj.Proj(proj='utm', zone=56, south=True, ellps='WGS84')
    # p = pyproj.Proj(proj='utm', zone=11, ellps='WGS84')
    # p = pyproj.Proj(proj='utm', zone=54, ellps='WGS84')

    return np.array([p(*c) for c in coords])

def get_coords(row):
    return np.array(row['geometry'].coords)

gdf['coords'] = gdf.apply(get_coords, axis=1)
gdf['coords_utm'] = gdf['coords'].apply(project_coords)

xi = np.array([c[0,0] for c in gdf['coords_utm']])[4:-1]
yi = np.array([c[0,1] for c in gdf['coords_utm']])[4:-1]

# vxi = np.array([c[0,0] for c in gdf['coords_utm']])[11]
# vyi = np.array([c[0,1] for c in gdf['coords_utm']])[11]

phi = gdf['orientation'].values[4:-1]
# vphi = gdf['orientation'].values[11]

# transform phi to float
phi = np.array([float(p) for p in phi])

# vphi = float(vphi)

from IHSetUtils import nauticalDir2cartesianDir, nauticalDir2cartesianDirP

phi = nauticalDir2cartesianDir(phi)
# vphi = nauticalDir2cartesianDirP(vphi)

# now we plot the transects 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))

for i, row in gdf.iterrows():
    ax.plot(row['coords_utm'][:,0], row['coords_utm'][:,1], color='black')
    ax.scatter(row['coords_utm'][0,0], row['coords_utm'][0,1], color='red')
    ax.scatter(row['coords_utm'][-1,0], row['coords_utm'][-1,1], color='blue')

    ax.scatter(xi, yi, color='black', s=100, marker='x' )

# ax.scatter(vxi, vyi, color='red', s=100, marker='x' )

plt.show()

import xarray as xr

ds = xr.Dataset({
    'hs': ('time', hs),
    'tp': ('time', tp),
    'dir': ('time', ddir),
    'obs': ('time_obs', obs_)},
    coords={'time': time,
            'time_obs': time_obs,})
            # 'vtime_obs': vtime_obs,})


ds.to_netcdf(r'/mnt/c/Users/freitasl/Documents/ReposGIT/metrics_for_y09_model/data/Angourie_daily.nc')
ds.close()