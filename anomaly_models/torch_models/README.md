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
from utilities.loaders.ucr import get_dataloaders

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
from utilities.loaders.ucr import get_dataloaders

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
from utilities.loaders.ucr import get_dataloaders

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
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=64, batch_size=64)

# Training  
model = CAE_M(window_size=64)
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## AE
[LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)

A simple LSTM based AutoEncoder.

sample usage:

```python
from anomaly_models.torch_models import AE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = AE(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## VAE
[A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder](https://arxiv.org/abs/1711.00614)

An LSTM based Variational AutoEncoder.

sample usage:

```python
from anomaly_models.torch_models import VAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = VAE(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```


## SAE
The Stationary AutoEncoder introduced by us.

sample usage:

```python
from anomaly_models.torch_models import SAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = SAE(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```


## FAE
The Fourier AutoEncoder applies the Fourier transform and is followed by an AutoEncoder, and it finally uses the Inverse Fourier Transform.

sample usage:

```python
from anomaly_models.torch_models import FAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = FAE(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## RD
The [Reverse Distillation](https://arxiv.org/pdf/2201.10703) method.

sample usage:

```python
from anomaly_models.torch_models import FAE, StudentDecoder
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training the teacher
teacher = FAE(window_size=256, device='cuda')  # or device='cpu'
teacher.learn(train_loader, n_epochs=10)

# Training the student
student = StudentDecoder(teacher_latent_dim=teacher.latent_dim)
student.learn(teacher, train_loader, n_epochs=10)
student.plot_results(teacher, test_loader)
```

## EBM
A very simple energy-based model that maps windows to energy values.

sample usage:

```python
from anomaly_models.torch_models import EBM
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = EBM(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## Wavelet Models
These models use the Wavelet Transform using [PyWavelets](https://pywavelets.readthedocs.io/).
The `WaveletAE` applies a number of transforms and uses convolutional layers.
The `EnergyBasedWavelet` applies a number of transforms and uses convolutional layers.

sample usage:

```python
from anomaly_models.torch_models import WaveletAE, EnergyBasedWavelet
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = WaveletAE(window_size=256, device='cuda')  # or EnergyBasedWavelet
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## Residual Learners
These models learn the difference of windows and reconstructed windows from other models.
Since other models might have different `forward` methods, we use a `recon_index` to retrieve the reconstructed windows of the base model.

### Residual EBM
The residual learner is another `EBM` here.

sample usage:

```python
from anomaly_models.torch_models import ResidualEBM, FAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Base Model
base = FAE(window_size=256, device='cuda')  # or device='cpu'

# Training  
model = ResidualEBM(window_size=256, device='cuda')
model.learn(train_loader, network=base, recon_index=1, n_epochs=10)
model.plot_results(test_loader, network=base, recon_index=1)
```

### Residual FAE
The residual learner is another `FAE` here.

sample usage:

```python
from anomaly_models.torch_models import ResidualFAE, FAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Base Model
base = FAE(window_size=256, device='cuda')  # or device='cpu'

# Training  
model = ResidualFAE(window_size=256, device='cuda')
model.learn(train_loader, network=base, recon_index=1, n_epochs=10)
model.plot_results(test_loader, network=base, recon_index=1)
```

## Twin
The window size will act as the larger frame for the _FAE_ and the `latent_dim` is the window size of a LSTM based _SAE_.

sample usage:

```python
from anomaly_models.torch_models import Twin
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=64)

# Training  
model = Twin(window_size=256, device='cuda')  # or device='cpu'
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

## Transformers

The Transformer version of AE, VAE, and SAE (so far).

```python
from anomaly_models.torch_models import TransformerAE, TransformerVAE, TransformerSAE
from utilities.loaders.ucr import get_dataloaders

# Dataset
dataset_path = '../../UCR/UCR2_preprocessed'
train_loader, test_loader = get_dataloaders(path=dataset_path, window_size=256, batch_size=16)

# Training  
model = TransformerAE(window_size=256, device='cuda')  # or TransformerVAE, or TransformerSAE
model.learn(train_loader, n_epochs=10)
model.plot_results(test_loader)
```

### Sparse Transformers
To make Transformers used in architectures sparse (similar to [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509))
you can simply perform the following steps on any of the Transformer based models and continue as usual.

```python
from anomaly_models.torch_models import TransformerAE, TransformerVAE, TransformerSAE
from anomaly_models.torch_models.SparseTransformer import replace_modules

# Instantiate original model  
model = TransformerAE(device='cuda') # or TransformerVAE, or TransformerSAE

# Replace Attention layers with Sparse Attention
replace_modules(model)

# If using cuda, you must move the new layers to cuda as well
model.to('cuda')
```

# References
- https://github.com/imperial-qore/TranAD/
