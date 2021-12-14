import torch, os
from torchvision.utils import save_image

from train_pl import VAE, GenerateCallback
from utils import visualize_manifold

pl_module = VAE.load_from_checkpoint("VAE_logs/lightning_logs/version_8520535/checkpoints/epoch=79-step=33759.ckpt")

img_grid = visualize_manifold(pl_module.decoder)
save_image(img_grid, 'vae_manifold.png', normalize=False)