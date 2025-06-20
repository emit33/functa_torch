# config.py
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import yaml
import torch
from dacite import from_dict, Config as DaciteConfig

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    width: int = 256
    depth: int = 5
    dim_in: int = 2
    dim_out: int = 3
    latent_dim: int = 64
    layer_sizes: Tuple[int, ...] = (256, 512)
    w0: float = 1.0
    modulate_scale: bool = True
    modulate_shift: bool = True
    device: torch.device = torch.device("gpu")


@dataclass
class TrainingConfig:
    latent_init_scale: float = 0.01
    latent_dim: int = 64
    outer_lr: float = 1e-4
    inner_lr: float = 1e-3
    l2_weight: float = 1e-6
    inner_steps: int = 10
    resolution: int = 256
    batch_size: int = 32
    n_epochs: int = 100


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "triangles"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig

    @classmethod
    def from_yaml(cls, yaml_path=None):
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "config.yaml"

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Configure dacite for type conversions
        dacite_config = DaciteConfig(
            type_hooks={
                Path: lambda x: PROJECT_ROOT / x if isinstance(x, str) else x,
                Tuple[int, ...]: lambda x: tuple(x) if isinstance(x, list) else x,
            }
        )

        return from_dict(data_class=cls, data=config_dict, config=dacite_config)
