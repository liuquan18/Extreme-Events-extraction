As more and more common the high frequency (like daily) data becomes avaiable, statistical analysis of **event**  (event statistics, e.g. the counts of specific extreme event) gets more and more popular than value statistics (e.g, mean and standard deviation). This blog shows how to (1) extract extreme events in time series, and (2) how to count events within a particular period.

# 1. Events extraction

 ## 1.1 Definition of events

**Positive (negative) events** are identified as those days at which the values are above (below) the thresholds. If the values of several consecutive days are above their corresponding shresholds, they are regarded as one single event, but with a duration of several days. Such thresholds can be quantile 90 (10) for extreme positive (negative) events, or mean plus (minus) one standard deviation for positive (negative) events. The quantile, mean and standard deviation should be calculated on each day of the year (so called climatological daily quantile, mean or std. can be easily implemented with cdo command: ```cdo -ydaypctl, -ydaymean, -ydaystd```, python details see below). Thus, we would have 365 different thresholds. See Fig.1 for details.

![img](https://pad.gwdg.de/uploads/065c5d6e50e2ebe4f42049f91.png)

> **Fig1.How to get threshold for an event.** Here shows the procedure to get the threshold for each *day of year* with grand-ensemble (simulation data) data. For each day, the threshold is got from a data with size of $year \times ens$ (black bold line outlined data). For the observational data, because there is only one realisation, for each day, one can get the threshold from  data of 7 days (window) ahead and after that day, and all the years, i.e., get the quantile or standard deviation from a data with size of $year\times window$. This can be easily implemented by *DataArray.rolling* function in python.

<img src="/Users/liuquan/Desktop/Screenshot 2022-02-23 at 16.42.49.png" alt="Screenshot 2022-02-23 at 16.42.49" style="zoom:50%;" />

> **Fig2. Definition of events.** Thresholds are calculated for each day of year. Here shows DJF only. 

## 1.2 Case study: positive (negative) NAO events

We show here the procedure to extract positive (negative) events of NAO and EA index gotten from ERA5. The data can be found here.

###  Data

```python
PCs = xr.open_dataset('/work/mh0033/m300883/task1/Hist_obs/obs/EOF_result/daily_index.nc')
PCs = PCs.daily_index
```

<img src="/Users/liuquan/Library/Application Support/typora-user-images/image-20220224151659317.png" alt="image-20220224151659317" style="zoom:50%;" />

The data contains two time series, which represent two teleconnection modes: NAO and EA.

```python
PCs.sel(mode = 'NAO').plot()
```

![PC_timeseries](/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/PC_timeseries.png)

![download](/Users/liuquan/Library/Mobile Documents/com~apple~CloudDocs/公众号/download.png)

The example time series are complicated in two aspects: data series are DJF only, which means data are not temporally continuous, and data in Dec of **last** year, and Jan and Feb in **this** year should be seen together as a independent time period, so we can not use  ``` groupby(year)``` directly; There are two series in the data, which can be easily popularized to spatio-temporal data with hundreds of time series.

## Events Extraction

**Whether the value is above (below) the threshold.** The funcition below can be adjusted to other thresholds like quantile 90. 

```python
def posExtValue(x):
    thr = x.mean(dim = 'time')+x.std(dim = 'time')
    return x-thr     # >0,  positive events
```

```python
def negExtValue(x):
    thr = x.mean(dim = 'time')-x.std(dim = 'time')
    return thr-x   # >0,  negative events
```

**Consecutivly above or below the threshold. **

This is the most complicated part of the method. What we want to do here is to find the groups of data records in which their values are all above (below) the threshold. For example, we have a dataset below, we want the records inside the red square be identified as indipendent events with different durations.

<img src="https://miro.medium.com/max/1330/1*frHPnaFOnq7j0KypRlGJng.png" alt="img" style="zoom: 67%;" />

> Fig3. consecutive records. [Source]( https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-certain-values-a6ed8e5d8cc) here. 

Although xarray are powerful to process spatial-temporal data, we don't find a good method to extract events. We solve the problem with pandas following a great tutorial [here](https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-certain-values-a6ed8e5d8cc). xarray DataArray can be easily converted to pandas DataFrame by ``` dataFrame=dataArray.to_dataframe().reset_index()```.



```python
def event_extract(PC,mode,extype):

  '''
  Extract events with different durations. Each event has an multi_index of mode, start_time, columns of duration, sum and mean.
  
  **Argument:**
  
  *PC*
  	A time series. here is the temporal index of NAO and EA.
  
  *mode*
  	'NAO' or 'EA'
  
  *extype*
  	extreme events type:[{'posEx':positive extreme events,'negEx':Negative extreme events]
  
  **Returns:**
  
  *Events*
  	multi_index of mode, start_time, columns of duration, sum and mean.
 
    '''
    
    if extype == 'posEx':
        ext_value = PC.sel(mode = mode).groupby(PC.time.dt.dayofyear).map(posExtValue)
    elif extype == 'negEx':
        ext_value = PC.sel(mode = mode).groupby(PC.time.dt.dayofyear).map(negExtValue)
    
    
    # for all ens
    events = list()
    

    # to dataframe
    exdf = ext_value.to_dataframe().reset_index() 

    # get consevutive days above/below threshold events groups
    for k, v in exdf[exdf['pseudo_pcs'] > 0].groupby((exdf['pseudo_pcs'] <= 0).cumsum()):

        # start_time and duration
        duration = pd.to_timedelta(v['time'].agg('size'),unit = 'D')
        start_time = v['time'].values[0]

        # data sum and mean as values
        data_single = np.array([duration,v['pseudo_pcs'].sum(),v['pseudo_pcs'].mean()])
        data_single.shape = (1,3)

        event_single = pd.DataFrame(
                                # data
                                data_single,

                                # columns
                                columns= ['duration','sum','mean'],

                                # multi-index 
                                index = pd.MultiIndex.from_product(
                                    [[mode],[start_time]],
                                    names = ['mode','start_time']
                                ))

        events.append(event_single)

    # events all ensembles.
    Events = pd.concat(events)

    return Events
```

