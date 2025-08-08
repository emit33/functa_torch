### Utils for my personal use, depending on my file structure. If this file is published publicly,  something has gone wrong.

import os
from pathlib import Path

import torch

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


def get_wdl(experiment_dir) -> tuple[int, int, int]:
    w_str, d_str, l_str = experiment_dir.name.split("_")[-3:]
    return (
        int(w_str),
        int(d_str),
        int(l_str),
    )
    # Written like this to satsify type checker


def get_final_checkpoint(experiment_dir):
    ckpt_names = os.listdir(experiment_dir / "ckpts")

    # Sort
    ckpt_names = sorted(
        ckpt_names, key=lambda x: int(x.split("_")[-1].removesuffix(".pth"))
    )

    if ckpt_names == []:
        return None
    else:
        return experiment_dir / "ckpts" / ckpt_names[-1]


def get_losses(experiment_dir):
    final_ckpt_path = get_final_checkpoint(experiment_dir)
    if final_ckpt_path is None:
        return None
    final_ckpt = torch.load(final_ckpt_path)

    return final_ckpt["avg_losses"]
