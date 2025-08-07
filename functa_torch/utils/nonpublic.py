### Utils for my personal use, depending on my file structure. If this file is published publicly,  something has gone wrong.

import os
from pathlib import Path

from functa_torch.utils.config import Config


def find_matching_experiments(experiment_parent, inds):
    # Convert string to path and int to list[int] if necessary
    if not isinstance(experiment_parent, Path):
        experiment_parent = Path(experiment_parent)

    if not isinstance(inds, list):
        inds = [inds]

    # list only sub‐dirs
    experiment_dirs = [
        experiment_parent / name
        for name in os.listdir(experiment_parent)
        if name[:3].isdigit()
    ]

    matched_dirs = [
        exp_dir for exp_dir in experiment_dirs if int(exp_dir.name[:3]) in inds
    ]

    return matched_dirs


def get_config_from_experiment_ind(experiment_ind):
    matched_experiment = find_matching_experiments(
        "/home/tempus/projects/functa_experiments", experiment_ind
    )
    assert len(matched_experiment) == 1, "Non-unique experiment index given"

    # Obtain config

    config_path = matched_experiment[0] / "config.yaml"
    config = Config.from_yaml(config_path)

    return config
