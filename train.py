import os

import hydra
from omegaconf import OmegaConf, DictConfig

import wandb
from pipeline.vae import main as vae_pipeline
from pipeline.vqvae import main as vqvae_pipeline
from pipeline.vae_ecg import main as vae_ecg_pipeline
from pipeline.autoencoder import main as autoencoder_pipeline
from pipeline.autoecgcoder import main as autoecgcoder_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Preparations
    os.makedirs(cfg.results_path, exist_ok=True)
    os.makedirs(cfg.checkpoint_path, exist_ok=True)

    wandb.init(
        project="autoencoders",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    if cfg.model.type == "AE":
        autoencoder_pipeline.main(cfg)

    if cfg.model.type == "ECG_AE":
        autoecgcoder_pipeline.main(cfg)

    if cfg.model.type == "VAE":
        vae_pipeline.main(cfg)

    if cfg.model.type == "VQ-VAE":
        vqvae_pipeline.main(cfg)

    if cfg.model.type == "ECG_VAE":
        vae_ecg_pipeline.main(cfg)


if __name__ == "__main__":
    main()
