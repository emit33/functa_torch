import dataclasses
import os
import shutil

import wandb

from functa_torch.utils.analysis import visualise_combined
from functa_torch.utils.nonpublic import get_config_from_experiment_ind
from functa_torch.utils.training import latentModulatedTrainer


def run_experiment(experiment_ind):
    config = get_config_from_experiment_ind(experiment_ind)

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

    wandb.finish()


if __name__ == "__main__":
    experiment_number = 133
    run_experiment(experiment_number)
