"""Run a functa training experiment from a YAML config file."""

import dataclasses
import os
import shutil
import argparse
from pathlib import Path
import wandb

from functa_torch.utils.config import Config
from functa_torch.utils.training import latentModulatedTrainer


def main() -> None:
    """Parse config path, initialize Weights & Biases, and run training."""
    # Obtain config path
    parser = argparse.ArgumentParser(
        description="Run functa training from a YAML config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
        dest="config_path",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Obtain config
    config = Config.from_yaml(cfg_path)

    # Clear checkpoints dir if it exists
    ckpt_dir = config.paths.checkpoints_dir
    if Path(ckpt_dir).exists():
        shutil.rmtree(ckpt_dir)

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

    wandb.finish()


if __name__ == "__main__":
    main()
