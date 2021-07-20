import hydra
from omegaconf import DictConfig
from src.experiment import Experiment


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print(config)
    exp = Experiment(config)


if __name__ == "__main__":
    main()
