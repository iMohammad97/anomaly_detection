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
```

## AE

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


## SAE

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

# References
- https://github.com/imperial-qore/TranAD/
