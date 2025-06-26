import os
import sys
from utils.config import Config
from utils.helpers import check_for_checkpoints
from utils.training import latentModulatedTrainer


if __name__ == "__main__":
    # Obtain config
    config = Config.from_yaml()

    # check_for_checkpoints(config.paths.checkpoint_dir)

    trainer = latentModulatedTrainer(config.model, config.training, config.paths).to(
        config.model.device
    )

    trainer.train()
