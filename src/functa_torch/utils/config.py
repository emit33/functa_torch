# config.py
from pathlib import Path
from dataclasses import asdict, dataclass, is_dataclass
from typing import Literal, Optional, Tuple
import yaml
import torch
from dacite import from_dict, Config as DaciteConfig


@dataclass
class ModelConfig:
    width: int = 256
    depth: int = 10
    dim_in: int = 2
    dim_out: int = 3
    latent_dim: int = 64
    layer_sizes: Optional[Tuple[int, ...]] = None
    w0: float = 1.0
    modulate_scale: bool = True
    modulate_shift: bool = True
    use_meta_sgd: bool = True
    final_activation: Optional[Literal["sigmoid"]] = None
    device: torch.device = torch.device("cuda")


@dataclass
class TrainingConfig:
    latent_init_scale: float = 0.01
    outer_lr: float = 1e-5
    inner_lr: float = 1e-2
    l2_weight: float = 1e-6
    inner_steps: int = 3
    batch_size: int = 16
    n_epochs: int = 500
    tensor_data: bool = True
    use_lr_schedule: bool = False
    normalise: Literal["01", "imagenet"] = "01"
    n_warmup_epochs: int = 500
    sample_prop: Optional[float] = None


@dataclass
class PathConfig:
    data_dir: Path
    checkpoints_dir: Path
    experiment_dir: Path
    figs_dir: Path
    ckpt_path: Optional[Path]


@dataclass
class OtherConfig:
    save_ckpt_step: Optional[int] = None
    save_figs: Optional[bool] = None


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig
    other: OtherConfig
    experiment_name: str

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Configure dacite for type conversions
        dacite_config = DaciteConfig(
            type_hooks={
                Path: lambda x: Path(x) if isinstance(x, str) else x,
                Tuple[int, ...]: lambda x: tuple(x) if isinstance(x, list) else x,
                torch.device: lambda s: torch.device(s) if isinstance(s, str) else s,
                float: lambda s: float(s) if isinstance(s, str) else s,
                int: lambda x: int(float(x)) if isinstance(x, (str, float)) else x,
            }
        )

        return from_dict(data_class=cls, data=config_dict, config=dacite_config)


def make_serializable(cfg):
    def convert(v):
        if is_dataclass(v) and not isinstance(v, type):
            return convert(asdict(v))
        if isinstance(v, dict):
            return {k: convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, torch.device):
            return str(v)  # e.g. 'cuda' / 'cpu'
        return v

    return convert(cfg)
