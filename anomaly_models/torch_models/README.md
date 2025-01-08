# List of (Working!) Models 

## DAGMM 
paper: [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://bzong.github.io/doc/iclr18-dagmm.pdf)
venue: ICLR 2018
code: 
sample usage:

```python
from anomaly_models.torch_models import DAGMM
from utilities.torch_ucr import get_dataloaders
# Dataset
train_loader, test_loader = get_dataloaders(path='../../UCR/UCR2_preprocessed', window_size=5, batch_size=64)
# Instantiate 
model = DAGMM()
# Train
model.train(train_loader)
# Evaluate
predictions = model.predict(test_loader)
```