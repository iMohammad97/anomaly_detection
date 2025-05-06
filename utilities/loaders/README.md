# Data Loaders for PyTorch Models

Here are a number of general parameters for all the loaders that you might need:
- `shuffle`: Setting this `True` will shuffle the training data for you.
- `step_size`: Altering this value from its default value, which is 1, returns the **training data** windows by taking larger steps.
- `seed`: You can set the seed by changing this parameter which is 0 by default.
- `train_split`: For the **NAB**, **Yahoo** and **TSB-AD-U** datasets, you can also select the percentage of training data.
- `normalize`: For the **NAB**, **Yahoo** and **TSB-AD-U** datasets, you can simply normalize them by setting `normalize=True`. The **TSB-AD-U** uses Z-normalization. 


## The UCR Anomaly Archive

You can load any of the 250 time series available by setting the `data_id` parameter.

```python
from utilities.loaders import get_ucr_loaders

path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_ucr_loaders(path=path, data_id=1, window_size=32, batch_size=256)
```

Alternatively, you can download the dataset using the second version of the loaders.

```python
from utilities.loaders import get_ucr2_loaders

train_loader, test_loader = get_ucr2_loaders(data_id=250, window_size=256)
```


## The Numenta Anomaly Benchmark (NAB)

There are 4 artificial and 3 real datasets here.
You must select one of the `possible_series` to load.

```python
from utilities.loaders import get_nab_loaders

path = '../../numenta'
possible_series = ['artificialWithAnomaly_art_daily_flatmiddle',
                   'artificialWithAnomaly_art_daily_jumpsdown',
                   'artificialWithAnomaly_art_daily_jumpsup',
                   'artificialWithAnomaly_art_increase_spike_density',
                   'realAWSCloudwatch_ec2_cpu_utilization_24ae8d',
                   'realKnownCause_ambient_temperature_system_failure',
                   'realTraffic_occupancy_6005']
ts = possible_series[0]
train_loader, test_loader = get_nab_loaders(path=path, time_series=ts, window_size=32, batch_size=256)
```


## The Yahoo S5 Labeled Anomaly Detection Dataset

There are four benchmarks in this dataset.
To select one, set the `benchmark` parameter to get the one you want.
The time series are a bit messy so to select the right `data_id` you must take a look at the dataset.

```python
from utilities.loaders import get_yahoo_loaders

path = '../../yahoo'
train_loader, test_loader = get_yahoo_loaders(path=path, benchmark=1, data_id=10, window_size=32, batch_size=256)
```



## TSB-AD-U
Taken from [🐘 The Elephant in the Room](https://thedatumorg.github.io/TSB-AD/).
You can load any of the 870 time series available by setting the `data_id` parameter.

```python
from utilities.loaders import get_tsb_loaders

train_loader, test_loader = get_tsb_loaders(data_id=870, window_size=256)
```
