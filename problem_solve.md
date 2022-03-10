A method to extract extreme events from multiple time series is presented in our last blog. However, There are still two issues:

1) ```groupby(time.dayofyear)``` in pandas works incorrectly when both non-leap year and leap year exists; 
2) The script is too time-consuming to be applied on spatial-temporal data.

# non-leap and leap year

Normally, we would delete data in `feb29` when calculate climitoligical daily means (means of multi years in each day). The problem is, even after deleting  `feb29`, the `dayofyear` is still different in non-leap year and leap year. The `dayofyear` attribute represents the "ordinal day" which in pandas (thus xarray) as "day since December 31st the preceding year". Therefore, in 2003, the 60th `dayofyear` is march 1st, while in 2004, the 60th `dayofyear` is February 29th. Then, if we `groupby(time.dayofyear)`, we would got 366 groups rather than expected 365 groups (note that all data in feb29 is removed already), and if we apply `mean` function on the groups, data in 2003.03.01 and 2004.02.29 will be aggravated together. 

One possible solution is, Instead of using `pandas.DatetimeIndex`, Using `xarray.CFTimeIndex`:

The easist way to solve this problem is:

1) Delete  `feb29` by either **cdo** command  `cdo -del29feb` or python **pandas** command `df_del = df[~((df.index.month==2) & (df.index.day==29))]` or **xarray** `arr.sel(time=~((arr.time.dt.month==2)&(arr.time.dt.day==29)))`

2) Replace the time index similar to the following commands:

   ```python
   index = xr.cftime_range('2002-01-01',periods = 3000,freq = '1D', calendar = 'noleap')
   index = index[~((index.month==2)&(index.day==29))]
   data['time']=index
   ```

   

# Accelerate the code by using groupby over mulitple columns

coming soon...
