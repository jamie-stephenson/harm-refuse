from harm_refuse import experiments
from harm_refuse.utils.config import Config

import hydra
from omegaconf import OmegaConf

@hydra.main(
    config_path="conf/experiments/",
    version_base=None,
)
def main(config: Config):
    print(config, type(config))
    run_experiment = getattr(experiments, config.name)
    run_experiment(config)
    

if __name__ == "__main__":
    main()
