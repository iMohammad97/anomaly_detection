# List of (Working!) Models 

## DAGMM 
[Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://bzong.github.io/doc/iclr18-dagmm.pdf)
ICLR 2018

sample usage:

```python
from anomaly_models.torch_models import DAGMM
from utilities.torch_ucr import get_dataloaders
# Dataset
train_loader, test_loader = get_dataloaders(path='../../UCR/UCR2_preprocessed', window_size=5, batch_size=64)
# Instantiate 
model = DAGMM()
# Train
model.train_model(train_loader, n_epochs=10)
# Evaluate
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
train_loader, test_loader = get_dataloaders(path='../../UCR/UCR2_preprocessed', window_size=5, batch_size=64)
# Instantiate 
model = MAD_GAN()
# Train
model.train_model(train_loader, n_epochs=10)
```