"""Helper utilities for navigating experiment directories"""

import os
from pathlib import Path
import torch

from functa_torch.utils.config import Config


def find_matching_experiments(
    experiment_parent: str | Path, inds: int | list[int]
) -> list[Path]:
    """Find experiment subdirectories whose names start with a 3-digit index in `inds`.

    Args:
        experiment_parent: Parent directory containing experiment subfolders.
        inds: A single index or a list of indices to match.

    Returns:
        List of matching experiment directories as Paths.
    """
    # Convert string to path and int to list[int] if necessary
    if not isinstance(experiment_parent, Path):
        experiment_parent = Path(experiment_parent)

    if not isinstance(inds, list):
        inds = [inds]

    # list only sub-dirs with leading 3-digit index
    experiment_dirs = [
        experiment_parent / name
        for name in os.listdir(experiment_parent)
        if name[:3].isdigit() and (experiment_parent / name).is_dir()
    ]

    matched_dirs = [
        exp_dir for exp_dir in experiment_dirs if int(exp_dir.name[:3]) in inds
    ]

    return matched_dirs


def get_config_from_experiment_ind(
    experiment_ind: int | list[int], experiment_root: str | Path
) -> Config:
    """Load Config for a given experiment index (or unique list with one index)."""
    matched_experiment = find_matching_experiments(experiment_root, experiment_ind)
    assert len(matched_experiment) == 1, "Non-unique experiment index given"

    # Obtain config

    config_path = matched_experiment[0] / "config.yaml"
    config = Config.from_yaml(config_path)

    return config


def get_wdl(experiment_dir: Path) -> tuple[int, int, int]:
    """Parse width, depth, latent_dim from the experiment directory name suffix."""
    w_str, d_str, l_str = experiment_dir.name.split("_")[-3:]
    return (
        int(w_str),
        int(d_str),
        int(l_str),
    )
    # Written like this to satsify type checker


def get_final_checkpoint(experiment_dir: Path) -> Path | None:
    """Return path to the final (highest-epoch) checkpoint in experiment_dir/ckpts, or None."""
    ckpts_dir = experiment_dir / "ckpts"
    if not ckpts_dir.is_dir():
        return None

    ckpt_names = os.listdir(ckpts_dir)
    # Sort by trailing epoch number
    ckpt_names = sorted(
        ckpt_names, key=lambda x: int(x.split("_")[-1].removesuffix(".pth"))
    )

    if ckpt_names == []:
        return None
    else:
        return ckpts_dir / ckpt_names[-1]


def get_losses(experiment_dir: Path) -> list[float] | None:
    """Load the 'avg_losses' list from the final checkpoint, or None if unavailable."""
    final_ckpt_path = get_final_checkpoint(experiment_dir)
    if final_ckpt_path is None:
        return None
    final_ckpt = torch.load(final_ckpt_path, map_location="cpu")

    return final_ckpt["avg_losses"]
