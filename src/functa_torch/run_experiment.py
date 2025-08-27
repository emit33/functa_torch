"""Run a functa training experiment by experiment index under a root directory."""

import dataclasses
import os
import shutil
from pathlib import Path
import wandb

from functa_torch.utils.directory_navigation import get_config_from_experiment_ind
from functa_torch.utils.training import latentModulatedTrainer


def run_experiment(experiment_ind: int, experiment_root: str | Path) -> None:
    """Resolve config by index, initialize W&B, and run training."""
    config = get_config_from_experiment_ind(experiment_ind, experiment_root)

    # Clear checkpoints dir if it exists
    ckpt_dir = Path(config.paths.checkpoints_dir)
    if ckpt_dir.exists():
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
    experiment_number = 133
    experiment_root = "/path/to/experiment/root"
    run_experiment(experiment_number, experiment_root)
