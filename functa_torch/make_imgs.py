from functa_torch.utils.analysis import visualise_loss, visualise_reconstructions
from functa_torch.utils.config import Config


if __name__ == "__main__":
    config = Config.from_yaml(
        "/home/tempus/projects/functa_experiments/21_mnist_wdl_32_10_16/config.yaml"
    )

    visualise_reconstructions(
        config.paths.checkpoints_dir,
        config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
    )
    visualise_loss(
        config.paths.checkpoints_dir,
        config.paths.figs_dir / (config.experiment_name + "_loss.png"),
    )
