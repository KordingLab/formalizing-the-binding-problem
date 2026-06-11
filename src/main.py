import hydra
from omegaconf import DictConfig
from trainer import Trainer


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    trainer = Trainer(cfg, output_dir=cfg.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
