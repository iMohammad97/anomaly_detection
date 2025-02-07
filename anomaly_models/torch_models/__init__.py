# TODO List:
# 	- Review code to make sure it works as intended
# 	- Make it possible to use GPU devices
# 	- Alter predict method to do what we might want
#   - Store losses separately based on their type (i.e. MSE and KLD in VAE)

# From papers (and TranAD repository)
from .dagmm import DAGMM
from .mad_gan import MAD_GAN
from .usad import USAD
from .cae_m import CAE_M

# Our AutoEncoder models
from .ae import AE
from .transformer_ae import TransformerAE
from .vae import VAE
from .transformer_vae import TransformerVAE
from .sae import SAE
from .transformer_sae import TransformerSAE
from .fae import FAE
from .dfae import DFAE # Denoising FAE

# Currently only works with FAE (and maybe DFAE)
from .rd import StudentDecoder

# Energy Based Models (only the first model is fully functional, the rest are in beta mode)
from .ebm import EBM
from .ebmper import PrioritizedEBM # EBM with PER (Prioritized Experience Replay)
from .ebmwld import EBMwLD # EBM with Langevin Dynamics

# Kolmogorov-Arnold Network AutoEncoder (in beta mode)
from .kan import KANAE

# Memory Based Models
from .mem_ae import MemAE # AE with Memory

