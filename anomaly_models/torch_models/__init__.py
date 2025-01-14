# TODO List:
# 	- Review code to make sure it works as intended
# 	- Make it possible to use GPU devices
# 	- Alter predict method to do what we might want
#   - Store losses separately based on their type (i.e. MSE and KLD in VAE)

from .dagmm import DAGMM
from .mad_gan import MAD_GAN
from .cae_m import CAE_M
from .ae import AE
from .vae import VAE
from .sae import SAE
