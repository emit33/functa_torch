import os
import shutil
from functa_torch.utils.config import Config
from functa_torch.utils.helpers import check_for_checkpoints
from functa_torch.utils.training import latentModulatedTrainer


def main():
    # Obtain config
    config = Config.from_yaml(
        "/home/tempus/projects/siren_analysis/functa_experiments/01_10_triangles_latent_64/config.yaml"
    )

    # check_for_checkpoints(config.paths.checkpoint_dir)
    if os.path.exists(config.paths.checkpoints_dir):
        shutil.rmtree(config.paths.checkpoints_dir)

    trainer = latentModulatedTrainer(config.model, config.training, config.paths).to(
        config.model.device
    )

    trainer.train()


if __name__ == "__main__":
    main()
