from harm_refuse import experiments
from harm_refuse.utils.config import Config

from hydra import compose, initialize_config_dir

from pathlib import Path

CONF = Path("conf")
EXPTS = (CONF / "experiments").resolve()

def main():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(EXPTS),
        job_name="harm_refuse"
    ):
        dictconfig = compose(
            config_name="cluster",
            overrides=[f"hydra.searchpath=[file://{CONF}]"]
        )

    config = Config.from_dictconfig(dictconfig)
    run_experiment = getattr(experiments, config.name)
    run_experiment(config)
    

if __name__ == "__main__":
    main()
