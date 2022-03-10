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
    thr = x.mean(dim = ('time','win'))+x.std(dim = ('time','win'))
    return x-thr     # >0,  positive events
```

```python
def negExtValue(x):
    thr = x.mean(dim =('time','win'))-x.std(dim = ('time','win'))
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

Our example time series is complicated in two aspects: data series are DJF only, which means data are not temporally continuous, and data in Dec of **last** year, and Jan and Feb in **this** year should be seen together as a independent time period, so we can not use `groupby(year)` directly, solution can be found below; There are two series in the data, which can be easily popularized to spatio-temporal data which contains hundreds of time series.

**Extract events by year** As stated before, our data is only for DJF, so we extract events sepertely for each year. In order to extract data properly, we offset the time series one month further with the function `pd.DateOffset(months=1)`.

```python
# xarray

def timeprocess(ts):
    ts_time = ts.time
    
    # roll forward 1 month to make the data into the same year
    ts_forward_time = pd.to_datetime(ts_time.values)+pd.DateOffset(months = 1) # +1 month
    ts_forward = ts.copy() 
    ts_forward['time'] = ts_forward_time
    
    # exclude the first and the last year where only two months of day avaiable.
    ts_out = ts_forward.sel(time = slice('1979','2020')) 
    return ts_out
```

**Multi time series** Because the return from function `getevents()` is a DataFrame, we can not use the `apply`function of pandas directly, we just use the `for` loop and `pd.concat()`. So the function can be easily adapted to spatial data.

```python
# xarray and pandas
def event_extract(ts,extype):

    # *extype:posEx;negEx
    # extreme value extraction
    
    # get threshold of one specific data from multi-years and 7 days before and after that day. 
    ts_roll = ts.rolling(time = 15,center = True).construct(window_dim = 'win')
    

    # contain all the events
    events_modes = list()

    # for all modes
    for mode in PCs.mode:
        
        # above/below the threshold
        if extype == 'posEx':
            ext_value = ts_roll.sel(mode = mode).groupby(ts_roll.time.dt.dayofyear).map(posExtValue)
        elif extype == 'negEx':
            ext_value = ts_roll.sel(mode = mode).groupby(ts_roll.time.dt.dayofyear).map(negExtValue)

        # reserve data in that day only.
        ext_value = ext_value.sel(windo = 0) 
        
        # process time
        ext_value = timeprocess(ext_value)
        
        # to dataframe
        df_ally = ext_value.to_dataframe().reset_index()
    
        # for all the years
        events_years = list()
        for year in df_ally.time.dt.year.unique():
            print(year)
            
            # select year
            df_singley=df_ally[df_ally.time.dt.year.eq(year)]
            
            # get events of this year
            events_singley = getevents(df_singley,'pcs')
            
            
            events_years.append(events_singley)
        Events_years = pd.concat(events_years).set_index('start_time')
        events_modes.append(Events_years)
        
    Events = pd.concat(events_modes,keys=PCs.mode.values,names = ['mode'])
    
    return Events
```

**Extracted Events in this example**

```python
posEvents = event_extract(PCs,'posEx')
negEvents = event_extract(PCs,'negEx')
Events = pd.concat([posEvents,negEvents],keys = ['positive','negative'],
                   names = ['event_type'])
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/m3WnZnGIHMiaXo1lcIvObGpvCdLmXBeEGMHHwiaCgqhhfpz6licKCavShVoQxaonWSAaw0ib07z8wkhNGms18PwqJA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

# 3. Statistics of events

As we have the extracted `Events`, we can do a lot of statistic analysis on this event series.

Below is a simple example to count the events by its `duration`.

```python
def count_by_duration(events):
    return events.groupby('duration').size()
def get_count(events):
    
    counts = list()
    # loop all the index other than start_time
    for extreme_type in events.index.levels[0]:
        for mode in events.index.levels[1]:

            # select data
            single_event = events.loc[idx[extreme_type,mode,:]]

            # counts on this data
            single_event_counts = count_by_duration(single_event)

            # index to construct df
            miindex = pd.MultiIndex.from_product([[extreme_type],
                                                  [mode],single_event_counts.index],
                                                  names = ['extreme_type','mode','duration'])

            single_event_counts_df = pd.DataFrame(single_event_counts.values,
                                                index = miindex,columns=['count']
                                                 )
            counts.append(single_event_counts_df)
    counts_df = pd.concat(counts)
    return counts_df
Event_counts = get_count(Events)
```

**Table**

```python
Event_counts_show = Event_counts.pivot_table(values = 'count',index = 'duration',columns = ['extreme_type','mode'])
Event_counts_show.index = [str(day)+' days' for day in Event_counts_show.index.days]

# last row for total
Event_counts_total = Event_counts_show.sum(axis = 0)
Event_counts_total.name = 'total'

Event_counts_show=Event_counts_show.append(Event_counts_total)
Event_counts_show
```

![图片](https://mmbiz.qpic.cn/mmbiz_png/m3WnZnGIHMiaXo1lcIvObGpvCdLmXBeEGpLZI32txkzqXoibw4YmTClT0XFMy8c4It3GEP8SPkx5HkrF6fKL8uBQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**Plot**

function to get relative frequency

```
def calprob(df):
    return df/df.sum()
fig,ax = plt.subplots(1,2,figsize = (13,6))
calprob(Event_counts_bar.loc['NAO'].loc['positive']).plot(kind='bar',ax=ax[0])
(-1*calprob(Event_counts_bar.loc['NAO'].loc['negative'])).plot(kind='bar',ax = ax[0],color = 'r')

calprob(Event_counts_bar.loc['EA'].loc['positive']).plot(kind='bar',ax=ax[1])
(-1*calprob(Event_counts_bar.loc['EA'].loc['negative'])).plot(kind='bar',ax = ax[1],color = 'r')

ax[0].set_title('mode = NAO')
ax[1].set_title('mode = EA')


ax[0].set_ylim(-0.26,0.26)
ax[1].set_ylim(-0.26,0.26)

ax[0].legend(['positive','negative'])
ax[1].legend(['positive','negative'])

plt.show()
```

![图片](https://mmbiz.qpic.cn/mmbiz_png/m3WnZnGIHMiaXo1lcIvObGpvCdLmXBeEGQLtThZyraX7HfMtCx5liaZuXgvfxHUXN8JuTLQeXBvAIdEziaGqHCJOg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

# Reference

[1]. https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-certain-values-a6ed8e5d8cc
