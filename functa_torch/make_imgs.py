from typing import Iterable, Union
from functa_torch.utils.analysis import visualise_combined
from functa_torch.utils.config import Config
from functa_torch.utils.nonpublic import (
    find_matching_experiments,
    get_config_from_experiment_ind,
)


def visualise_experiments_from_indices(experiment_inds: Union[int, Iterable]):
    if isinstance(experiment_inds, int):
        experiment_inds = [experiment_inds]

    for ind in experiment_inds:
        config = get_config_from_experiment_ind(ind)

        try:
            visualise_combined(
                config.paths.checkpoints_dir,
                config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
            )

        except (FileNotFoundError, RuntimeError) as e:
            # only catch the errors we expect, and show what went wrong
            print(f"{config.experiment_name} failed: {e}")


if __name__ == "__main__":
    experiment_numbers: Union[int, Iterable] = range(35, 42)

    visualise_experiments_from_indices(experiment_numbers)
