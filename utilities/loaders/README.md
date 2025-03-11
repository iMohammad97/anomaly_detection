# Data Loaders for PyTorch Models

Here are a number of general parameters for all the loaders that you might need:
- `shuffle`: Setting this `True` will shuffle the training data for you.
- `step_size`: Altering this value from its default value, which is 1, returns the **training data** windows by taking larger steps.
- `seed`: You can set the seed by changing this parameter which is 0 by default.

## The UCR Anomaly Archive

You can load any of the 250 time series available by setting the `data_id` parameter.

```python
from utilities.loaders import get_ucr_dataloaders

dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_ucr_dataloaders(path=dataset_path,  data_id=1, window_size=32, batch_size=256)
```


## The Yahoo S5 Labeled Anomaly Detection Dataset

There are four benchmarks in this dataset.
To select one, set the `benchmark` parameter to get the one you want.
The time series are a bit messy so to select the right `data_id` you must take a look at the dataset.

```python
from utilities.loaders import get_yahoo_dataloaders

dataset_path = '../../yahoo'
train_loader, test_loader = get_yahoo_dataloaders(path=dataset_path, benchmark=1,  data_id=10, window_size=32, batch_size=256)
```