from pytorch_lightning.trainer import trainer
import torch
from train_pl import VAE, GenerateCallback

pl_module = VAE.load_from_checkpoint("VAE_logs/lightning_logs/version_8519982/checkpoints/epoch=6-step=2953.ckpt")

multichannel_samples = pl_module.sample(4)
B, C, H, W = multichannel_samples.shape
samples = torch.zeros((B, 1, H, W)).to(pl_module.device)

for h in range(H):
    for w in range(W):
        samples[:, 0, h, w] = (torch.multinomial(multichannel_samples[:, :, h, w].softmax(dim=1), 1)).view(-1)

samples = samples / 15