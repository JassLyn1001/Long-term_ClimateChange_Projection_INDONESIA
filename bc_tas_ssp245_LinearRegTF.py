import numpy as np
import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt
import os,sys
import cftime
import dask
import warnings
from glob import glob
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
import xesmf as xe
import gc

### -------------------------------------------------------------

# Configuration
PATHS = {
    'cmip6': "/raid/nfs_storageIPB/NC4_WorkDir/datasets/NEX-GDDP-CMIP6_GLOBAL/",
    'obs': "/raid/nfs_storageIPB/NC4_WorkDir/datasets/OBS/ERA5Land/Tmean/"}

VAR = 'tas'
EXPERIMENTS = ['historical', 'ssp245'] # 

def get_sorted_files(model, experiment):
    """Get sorted list of files for a given model and experiment."""
    pattern = f"{PATHS['cmip6']}{model}/{experiment}/{VAR}_sea/seasia_{VAR}_*_{model}_{experiment}_*.nc"
    return sorted(glob(pattern)) 

def get_sorted_files_robust(model, experiment):
    """Get sorted list of files with directory existence checks."""
    base_path = f"{PATHS['cmip6']}{model}/{experiment}/"
    new_path = f"/raid/nfs_storageIPB/NC4_WorkDir/datasets/scripts/NEX-GDDP-CMIP6_SEA/{model}/{experiment}/"
    
    # Check if the base directory exists
    if not os.path.exists(base_path):
        print(f"Base directory does not exist: {base_path}")
        return []
    
    # Try seasia_{VAR} directory first
    sea_dir = os.path.join(base_path, f"{VAR}_sea")
    if os.path.exists(sea_dir):
        pattern_sea = os.path.join(sea_dir, f"seasia_{VAR}_*_{model}_{experiment}_*.nc")
        files = sorted(glob(pattern_sea))
        if files:
            return files
    
    # Try regular tas directory in ../scripts/..
    regular_dir = os.path.join(new_path, VAR)
    if os.path.exists(regular_dir):
        pattern_regular = os.path.join(regular_dir, f"{VAR}_*_{model}_{experiment}_*.nc")
        files = sorted(glob(pattern_regular))
        if files:
            return files
    
    return []

# Begin to get file passed in bash script
if len(sys.argv) > 1:
    model = sys.argv[1] # sys.argv[0] return file name of the script
    print(f"Processing: {model}")

    # Get all experiment files in one dictionary comprehension
    experiment_files = {
        exp: get_sorted_files(model, exp)
        for exp in EXPERIMENTS
    }

    # Unpack to individual variables
    hist_path = experiment_files[EXPERIMENTS[0]]
    scenario_path = experiment_files[EXPERIMENTS[1]]

    # Function to read in netcdf files
    def open_mfile(filapaths): # read multiple files
        with xr.open_mfdataset(filapaths, chunks=dict(time=365)) as ds:
            print(ds.keys())
        return ds

    def open_file(filapaths): # read single files
        with xr.open_dataset(filapaths) as ds:
            print(ds.keys())
        return ds

    hist_ds = open_mfile(hist_path)
    scenario_1_ds = open_mfile(scenario_path) # 2015-2100 read in all timesteps of model data 
    obs_ds = open_file(PATHS['obs']+'dailymean_era5land_t2m_id_degCelcius_1981-2021.nc')
                                    
    # Define the target lat-lon grid based on observation data
    ds_target = xr.Dataset(
        {
            "lat": (("lat"), obs_ds.latitude.values), # Y
            "lon": (("lon"), obs_ds.longitude.values), # X
        }
    )

    def regrid_with_dask(ds, target_grid, method="nearest_s2d"):
        """Perform regridding in parallel using Dask."""
        regridder = xe.Regridder(
            ds,
            target_grid,
            method,
        )
        return regridder(ds).compute()  # Trigger parallel computation

    # Perform parallel regridding
    hist_ds_regridded = regrid_with_dask(hist_ds, ds_target) # ignore UserWarning
    scenario_1_regridded = regrid_with_dask(scenario_1_ds, ds_target)

    ### Calendar & Unit Conversion
    # Get length of sample historical period for each data (obs and model)
    obs_length = len(obs_ds.sel(valid_time=slice("1991","2014")).valid_time.values)
    model_length = len(hist_ds_regridded.sel(time=slice("1991", "2014")).time.values)

    # Baseline
    if model_length != obs_length: # proceed conversion if model's calendar is not equal to standard or proleptic gregorian
        print("The calendar is not standard. Starting calendar conversion for baseline period...")
        # Convert calendar to standard/proleptic gregorian
        hist_std_da = hist_ds_regridded.tas.convert_calendar(
            "proleptic_gregorian", dim='time', align_on="year", missing=np.nan)
        # Interpolate missing value between dates after converting calendar
        hist_std_da = hist_std_da.interpolate_na(dim="time") - 273.15 # convert unit to degree Celcius
        print(f"Calendar conversion for {model} is done")
    else:
        # Don't do calendar conversion and then convert the unit
        hist_std_da = hist_ds_regridded.tas - 273.15
        print("The calendar system is standard")

    # Future
    if model_length != obs_length: # proceed conversion if model's calendar is not equal to standard or proleptic gregorian
        print("The calendar is not standard. Starting calendar conversion for future period... ")
        # Convert calendar to standard/proleptic gregorian
        scenario_1_std_da = scenario_1_regridded.tas.convert_calendar(
            "proleptic_gregorian", dim='time', align_on="year", missing=np.nan)

        # Interpolate missing value between dates after converting calendar
        scenario_1_std_da = scenario_1_std_da.interpolate_na(dim="time") - 273.15 
        print(f"Calendar conversion for {model} is done")
    else:
        # Don't do calendar conversion and just define xarray.dataset as xarray.dataarray then convert to mm/day
        scenario_1_std_da = scenario_1_regridded.tas - 273.15
        print("The calendar system is standard")

    ### Group by month
    # Rename lat lon dimensions of observation dataset and select time dimension for baseline period
    obs_da = obs_ds.t2m.rename({'latitude':'lat', 'longitude':'lon'})
    sel_obs_da = obs_da.sel(valid_time=slice('1991','2020'))

    ### Data selection
    # Combine historical data with early period of future data
    hist_da = xr.concat([hist_std_da.sel(
        time=slice('1991','2014')),scenario_1_std_da.sel(time=slice('2015','2020'))],dim='time')
    future_1_da = scenario_1_std_da.sel(time=slice('2021','2100'))

    # Group data by each month
    obs_by_month = sel_obs_da.groupby("valid_time.month")
    mod_by_month = hist_da.groupby("time.month")
    mod_by_month_future_1 = future_1_da.groupby("time.month")

    ### Bias-Correction (stats.rv_histogram) for Baseline
    # Initialize models once
    linreg_model = LinearRegression()

    # Define a function for bias-correction
    def biascorrection_baseline(obs, base, linreg_model):
        # Check for NaN values over ocean area
        mod_no_nan = base[~np.isnan(base)] # get values except NaN from model
        obs_no_nan = obs[~np.isnan(obs)] # get values except NaN from observation
        N_obs = len(obs_no_nan)
        N_mod = len(mod_no_nan)

        length_obs = len(obs)
        length_mod = len(base)
        if (N_mod!=0 and N_obs!=0) and (N_mod==length_mod and N_obs==length_obs) :
            # Proceed if grids located over the land and there is no nan value on the land grid point
            # Bias-correction based on histogram of the data
            obs_hist = np.histogram(obs, bins=500)
            obs_dist = stats.rv_histogram(obs_hist, density=True)
            mdl_hist = np.histogram(base, bins=500)
            mdl_dist = stats.rv_histogram(mdl_hist, density=True)
            
            # Inverse CDF calculation
            prob_grid = np.linspace(0, 1, 101)
            obs_invcdf = obs_dist.ppf(prob_grid)
            mdl_invcdf = mdl_dist.ppf(prob_grid)
            
            # Find indices where both arrays have non-NaN values
            valid_indices = ~np.isnan(obs_invcdf) & ~np.isnan(mdl_invcdf)
            obs_invcdf = obs_invcdf[valid_indices]
            mdl_invcdf = mdl_invcdf[valid_indices]

            # Fit the model
            mdl_to_train = mdl_invcdf.reshape(-1,1)
            linreg_model.fit(mdl_to_train, obs_invcdf)

            # Correcting baseline data
            mdl_to_correct = np.array(base).reshape(-1,1)
            mdl_corrected = linreg_model.predict(mdl_to_correct)  # get the predicted response
            return mdl_corrected
        else:
            out = np.empty(len(base))
            out[:] = np.nan
            return out #if the data point is in ocean area than return np.nan for output

    # Perform bias-correction for baseline period
    #np.seterr(divide='ignore', invalid='ignore')
    corrected_baseline = {}
    for month in range(1, 13):
        # Get the data for current month
        baseline_month = mod_by_month[month]
        obs_month = obs_by_month[month]

        # Apply bias correction
        corrected_data = xr.apply_ufunc(
            biascorrection_baseline,
            obs_month,
            baseline_month,
            linreg_model,
            input_core_dims=[["valid_time"], ["time"], []],
            output_core_dims=[['time']],
            exclude_dims={'time'},
            vectorize=True,
            output_dtypes=[np.float32])

        # Restore the time coordinates
        corrected_data['time']= baseline_month.time
        corrected_baseline[month] = corrected_data.transpose('time', ...)
        print(f'Bias Correction for month: {month} is done')
    
    del corrected_data
    gc.collect()
    
    ### Save bias-corrected data to new netcdf files 
    print(f'Save bias-corrected data to new netcdf files')
    bc_baseline_month = []
    for month in range(1,13):
        # unpack grouped by month data
        bc_baseline_month.append(corrected_baseline[month])

    # Concatenate month
    bc_baseline_da = xr.concat(bc_baseline_month, dim=('time')).sortby('time')

    # Get attributes from all coordinate dimensions
    time_att = hist_ds.time.attrs 
    lon_att = hist_ds.lon.attrs
    lat_att = hist_ds.lat.attrs

    # Define a function to assign dataarray attributes for bias-corrected data
    def assign_dim_attrs(da):
        da['time'] = da.time.assign_attrs(time_att)
        da['lat'] = da.lat.assign_attrs(lat_att)
        da['lon'] = da.lon.assign_attrs(lon_att)
        return da

    # Define dataarray's attributes
    da_attrs = {
        'standard_name': 'air_temperature',
        'long_name': 'Daily Near-Surface Air Temperature',
        'units': 'degC',
        }

    ds_attrs = {
        'title': 'Statistical Downscaling output based on Bias-Correction approach prepared for Indonesia Fourth National Communication (NC4)',
        'authors': 'Akhmad Faqih; Jassica Listyarini; Syamsu Dwi Jadmiko',
        'institution': 'Department of Geophysics and Meteorology, Faculty of Mathematics and Natural Sciences, IPB University; Centre for Climate Risk and Opportunity Management in Southeast Asia Pasific (CCROM - SEAP), IPB University',
        'table_id': 'day',
        'variable_id': f'{VAR}',

    }

    # Assign attributes for bias-corrected baseline data
    bc_dim_attrs = assign_dim_attrs(bc_baseline_da)
    bc_da_attrs = bc_dim_attrs.assign_attrs(da_attrs)
    new_ds = bc_da_attrs.to_dataset(dim=None, name=f'{VAR}')
    bc_baseline_ds= new_ds.assign_attrs(ds_attrs)

    # Define file's configurations
    base_period = '19910101-20201231'
    future_period_1 = '20210101-21001231'
    future_EXPERIMENTS = EXPERIMENTS[1]
    table_id = 'day'
    
    # Create output directory
    outdir_dir = f"/raid/nfs_storageIPB/NC4_WorkDir/datasets/PROCESSED/{VAR}_linreg/"
    path = Path(outdir_dir)/model
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{model}' is created or already exists")
    except Exception as e:
        print(f"Failed to create directory: {str(e)}")
        raise

    def save_dataset(ds, experiment_suffix):
        """Helper function to save datasets into netcdf format with error handling"""
        try:
            filename = f"bias-corrected_{VAR}_id_{table_id}_{model}_{EXPERIMENTS[0]}_{experiment_suffix}_{base_period}.nc"
            output_path = path / filename
            # Add another attributes
            ds.attrs['experiment_id'] = EXPERIMENTS[0]
            ds.attrs['frequency'] = table_id
            ds.attrs['mip_era'] = 'CMIP6'
            ds.attrs['source_id'] = model
            print(f"Attempting to save to: {output_path}")
            # Load and write in one operation to minimize memory usage
            ds.to_netcdf(
                output_path,
                format="NETCDF4",
                engine="netcdf4",
                encoding={VAR: {"dtype": "float32"}},  # Use variable name from VAR
                unlimited_dims='time'
            )
            print(f"Successfully saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")
            raise

    def save_fdataset(ds, experiment_suffix, f_period):
        """Helper function to save future datasets into netcdf format with error handling"""
        try:
            filename = f"bias-corrected_{VAR}_id_{table_id}_{model}_{experiment_suffix}_{f_period}.nc"
            output_path = path / filename
            # Add another attributes
            ds.attrs['experiment_id'] = experiment_suffix
            ds.attrs['frequency'] = table_id
            ds.attrs['mip_era'] = 'CMIP6'
            ds.attrs['source_id'] = model
            print(f"Attempting to save to: {output_path}")
            # Load and write in one operation to minimize memory usage
            ds.to_netcdf(
                output_path,
                format="NETCDF4",
                engine="netcdf4",
                encoding={VAR: {"dtype": "float32"}},  # Use variable name from VAR
                unlimited_dims='time'
            )
            print(f"Successfully saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")
            raise

    # Save bias-corrected data to new netcdf file
    save_dataset(bc_baseline_ds, future_EXPERIMENTS)

    ### Bias-Correction (stats.rv_histogram) for Future period
    def biascorrection_future(obs, base, future, linreg_model):
        # Check for NaN values over ocean area
        mod_no_nan = base[~np.isnan(base)] # get values except NaN from model
        obs_no_nan = obs[~np.isnan(obs)] # get values except NaN from observation
        N_obs = len(obs_no_nan)
        N_mod = len(mod_no_nan)

        length_obs = len(obs)
        length_mod = len(base)
        if (N_mod!=0 and N_obs!=0) and (N_mod==length_mod and N_obs==length_obs) :
            # Bias-correction based on histogram of the data
            obs_hist = np.histogram(obs, bins=500)
            obs_dist = stats.rv_histogram(obs_hist, density=True)
            mdl_hist = np.histogram(base, bins=500)
            mdl_dist = stats.rv_histogram(mdl_hist, density=True)

            # Inverse CDF calculation
            prob_grid = np.linspace(0, 1, 101)
            obs_invcdf = obs_dist.ppf(prob_grid)
            mdl_invcdf = mdl_dist.ppf(prob_grid)

            # Find indices where both arrays have non-NaN values
            valid_indices = ~np.isnan(obs_invcdf) & ~np.isnan(mdl_invcdf)
            obs_invcdf = obs_invcdf[valid_indices]
            mdl_invcdf = mdl_invcdf[valid_indices]

            # Fit the model
            mdl_to_train = mdl_invcdf.reshape(-1,1)
            linreg_model.fit(mdl_to_train, obs_invcdf)

            # Correcting future data
            mdl_to_correct = np.array(future).reshape(-1,1)
            mdl_corrected = linreg_model.predict(mdl_to_correct) # get the predicted response
            return mdl_corrected
        else:
            out = np.empty(len(future))
            out[:] = np.nan
            return out #if the data point is in ocean area than return np.nan for output


    # Perform bias-correction for 2021-2060 period
    corrected_future_1 = {}
    for month in range(1, 13):
        # Get the data for current month
        baseline_month = mod_by_month[month]
        obs_month = obs_by_month[month]
        future_month = mod_by_month_future_1[month]

        # Apply bias correction
        corrected_data = xr.apply_ufunc(
            biascorrection_future,
            obs_month,
            baseline_month,
            future_month,
            linreg_model,
            input_core_dims=[["valid_time"], ["time"], ["time"], []],
            output_core_dims=[['time']],
            exclude_dims={'time'},
            vectorize=True,
            output_dtypes=[np.float32])

        # Restore the time coordinates
        corrected_data['time']= future_month.time
        corrected_future_1[month] = corrected_data.transpose('time', ...)
        print(f'Bias Correction for month: {month} is done')

    del corrected_data
    gc.collect()

    bc_future_month_1 = []
    for month in range(1,13):
        # unpack grouped by month data
        bc_future_month_1.append(corrected_future_1[month])

    # Concatenate month
    bc_future_1_da = xr.concat(bc_future_month_1, dim=('time')).sortby('time')

    # Assign attributes for bias-corrected future data
    bc_future_dim_attrs_1 = assign_dim_attrs(bc_future_1_da)
    bc_future_da_attrs_1 = bc_future_dim_attrs_1.assign_attrs(da_attrs)
    future_1_new_ds = bc_future_da_attrs_1.to_dataset(dim=None, name=f'{VAR}')
    bc_future_1_ds= future_1_new_ds.assign_attrs(ds_attrs)

    # Exporting future bias-corrected data
    save_fdataset(bc_future_1_ds, future_EXPERIMENTS, future_period_1)

    print(f"Bias-Correction for {model} is done")
else:
    print("No argument/model's name provided. Bias-correction can't be started")
