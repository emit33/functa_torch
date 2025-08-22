import dataclasses
import os
import shutil
import argparse

import wandb

from functa_torch.utils.analysis import visualise_combined
from functa_torch.utils.config import Config
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

    if os.path.exists(config.paths.checkpoints_dir):
        shutil.rmtree(config.paths.checkpoints_dir)

    run_name = f"{config.experiment_name}_job{os.environ.get('SLURM_JOB_ID','local')}"
    wandb.init(
        project="functa",
        name=run_name,
        config=dataclasses.asdict(config),
    )

    trainer = latentModulatedTrainer(
        config.model, config.training, config.paths, config.other
    ).to(config.model.device)

    trainer.train()

    # Create visualisations
    if config.paths.figs_dir is not None:
        visualise_combined(
            config.paths.checkpoints_dir,
            config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
        )

    wandb.finish()


if __name__ == "__main__":
    main()
