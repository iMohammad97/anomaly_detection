# Torch Models
Models in this package share similar methods and are all implemented using PyTorch.

## Guide
Here are some useful tips as to how you can work with these models:
- To **train** a model, just use `model.learn(train_loader, n_epochs)`
- To get the reconstructions use `model.predict(test_loader)`
- To **plot** the reconstructions next to the original time series and the reconstruction losses use `model.plot_results(test_loader)`
- To plot the training **losses** use `model.plot_losses()`
- To **save** a model use `model.save(path)`
- To **load** a model use `model = Model_Class.load(path)`

In these examples `train_loader` and `test_loader` are PyTorch `DataLoader`s.
The `plot_losses` method can also take a `fig_size` input which specifies the width and height of the plot.
If you omit the `path` input for saving, the default path will be models name and the number of epochs you trained it for.

**Warning**: Evidently google colab doesn't show plotly plots when you print something in the same cell. 
If you don't see anything after calling `plot_results` for your model, try running it in a new empty cell.

# List of (Working!) Models 

## DAGMM 
[Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://bzong.github.io/doc/iclr18-dagmm.pdf)
ICLR 2018

sample usage:

```python
from anomaly_models.torch_models import DAGMM
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=5, batch_size=64)

# Training 
model = DAGMM()
model.learn(train_loader, n_epochs=10)
predictions = model.predict(test_loader)
```

## MAD-GAN
[MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997)
ICANN 2019

sample usage:

```python
from anomaly_models.torch_models import MAD_GAN
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=5, batch_size=64)

# Training 
model = MAD_GAN()
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## USAD
[USAD : UnSupervised Anomaly Detection on Multivariate Time Series](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392)
KDD 20

sample usage:

```python
from anomaly_models.torch_models import USAD
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=5, batch_size=64)

# Training  
model = USAD()
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## CAE-M
[Unsupervised Deep Anomaly Detection for Multi-Sensor Time-Series Signals](https://arxiv.org/pdf/2107.12626)
TKDE 2021

sample usage:

```python
from anomaly_models.torch_models import CAE_M
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=64, batch_size=64)

# Training  
model = CAE_M(window_size=64)
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## AE
A simple LSTM based AutoEncoder.

sample usage:

```python
from anomaly_models.torch_models import AE
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = AE(window_size=256, device='cuda') # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## VAE
An LSTM based Variational AutoEncoder.

sample usage:

```python
from anomaly_models.torch_models import VAE
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = VAE(window_size=256, device='cuda') # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```


## SAE
The Stationary AutoEncoder introduced by us.

sample usage:

```python
from anomaly_models.torch_models import SAE
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = SAE(window_size=256, device='cuda') # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```


## FAE
The Fourier AutoEncoder applies the Fourier transform and is followed by an AutoEncoder, and it finally uses the Inverse Fourier Transform.

sample usage:

```python
from anomaly_models.torch_models import FAE
from utilities.torch_ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = FAE(window_size=256, device='cuda') # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

# References
- https://github.com/imperial-qore/TranAD/
