"""Batch visualization helper to render reconstructions and loss plots for experiments."""

from typing import Iterable, Union, Sequence
from pathlib import Path
from functa_torch.utils.analysis import visualise_combined
from functa_torch.utils.config import Config
from functa_torch.utils.directory_navigation import (
    get_config_from_experiment_ind,
)


def visualise_experiments_from_indices(
    experiment_inds: Union[int, Iterable[int]],
    resolution: Sequence[int],
    experiment_root: Union[str, Path],
) -> None:
    """Visualize reconstructions and loss for one or more experiment indices.

    Args:
        experiment_inds: Single index or an iterable of indices.
        resolution: Target reconstruction resolution, e.g. (H, W).
        experiment_root: Root directory containing experiment folders.
    """
    if isinstance(experiment_inds, int):
        experiment_inds = [experiment_inds]

    for ind in experiment_inds:
        config: Config = get_config_from_experiment_ind(ind, experiment_root)

        try:
            visualise_combined(
                config.paths.checkpoints_dir,
                config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
                resolution,
            )

        except (FileNotFoundError, RuntimeError) as e:
            print(f"{config.experiment_name} failed: {e}")


if __name__ == "__main__":
    experiment_numbers = range(93, 111)
    resolution = (64, 64)
    experiment_root = "/path/to/experiment/root"

    visualise_experiments_from_indices(experiment_numbers, resolution, experiment_root)
