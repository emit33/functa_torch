import os
import shutil

import numpy as np
from functa_torch.utils.analysis import visualise_loss, visualise_reconstructions
from functa_torch.utils.config import Config
from functa_torch.utils.training import latentModulatedTrainer


def main():
    # Obtain config
    config = Config.from_yaml(
        "/home/tempus/projects/functa_experiments/00_test_config/config.yaml"
    )

    if os.path.exists(config.paths.checkpoints_dir):
        shutil.rmtree(config.paths.checkpoints_dir)

    trainer = latentModulatedTrainer(
        config.model, config.training, config.paths, config.other
    ).to(config.model.device)

    trainer.train()

    # Create visualisations
    if config.paths.figs_dir is not None:
        visualise_reconstructions(
            config.paths.checkpoints_dir,
            config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
        )
        visualise_loss(
            config.paths.checkpoints_dir,
            config.paths.figs_dir / (config.experiment_name + "_loss.png"),
        )


if __name__ == "__main__":
    main()
