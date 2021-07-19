import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print(config)


if __name__ == "__main__":
    main()
