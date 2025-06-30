import os
import shutil
import argparse

import yaml
from functa_torch.utils.config import Config
from functa_torch.utils.helpers import check_for_checkpoints
from functa_torch.utils.training import latentModulatedTrainer


def main():
    # Obtain config path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
        dest="config_path",
    )
    args = parser.parse_args()

    # Obtain config
    config = Config.from_yaml(args.config_path)

    # check_for_checkpoints(config.paths.checkpoint_dir)
    if os.path.exists(config.paths.checkpoints_dir):
        shutil.rmtree(config.paths.checkpoints_dir)

    trainer = latentModulatedTrainer(config.model, config.training, config.paths).to(
        config.model.device
    )

    trainer.train()


if __name__ == "__main__":
    main()
