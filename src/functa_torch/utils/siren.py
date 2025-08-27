"""SIREN layers and latent-modulated SIREN model with FiLM-based modulations."""

from typing import Callable, Dict, Literal, Optional, Tuple
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class MetaSGDLrs(nn.Module):
    """
    Container for per-parameter learning rates used by Meta-SGD.

    Notes:
    - This module only stores learnable LR parameters (nn.Parameter) so they are
      optimized with the outer optimizer just like other model weights.
    """

    def __init__(
        self,
        num_lrs: int,
        lrs_init_range: Tuple[float, float] = (0.005, 0.1),
        lrs_clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        """
        Args:
            num_lrs: Number of learning rates to learn (equals latent_dim).
            lrs_init_range: Uniform range for initializing the LRs.
            lrs_clip_range: Suggested clipping range for LRs (not applied here).
        """
        super().__init__()
        self.num_lrs = num_lrs
        self.lrs_init_range = lrs_init_range
        self.lrs_clip_range = lrs_clip_range

        # Initialize learning rates ~ U[lrs_init_range[0], lrs_init_range[1]]
        meta_sgd_lrs = (
            torch.rand(num_lrs) * (lrs_init_range[1] - lrs_init_range[0])
            + lrs_init_range[0]
        )
        self.meta_sgd_lrs = nn.Parameter(meta_sgd_lrs)


class Sine(nn.Module):
    """Applies a scaled sine transform to input: out = sin(w0 * in)."""

    def __init__(self, w0: float = 1.0):
        """
        Args:
            w0: Scale factor (omega_0) in the SIREN activation.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


class FiLM(nn.Module):
    """
    FiLM modulation: out = scale * in + shift.

    Notes:
      We currently initialize FiLM layers as the identity. However, this may not
      be optimal. In pi-GAN for example they initialize the layer with a random
      normal.
    """

    def __init__(
        self, dim_in: int, modulate_scale: bool = True, modulate_shift: bool = True
    ):
        """
        Args:
            dim_in: Feature dimension to modulate.
            modulate_scale: Whether to learn multiplicative modulation.
            modulate_shift: Whether to learn additive modulation.
        """
        super().__init__()
        self.dim_in = dim_in

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # Initialise as identity; make parameters trainable iff modulation is enabled.
        self.scale = 1.0
        self.shift = 0.0

        if self.modulate_scale:
            self.scale = nn.Parameter(torch.ones(dim_in))

        if self.modulate_shift:
            self.shift = nn.Parameter(torch.zeros(dim_in))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Module):
    """
    A single SIREN layer with optional FiLM modulation and optional sine activation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        is_first: bool = False,
        is_last: bool = False,
        modulate_shift: bool = True,
        modulate_scale: bool = True,
        apply_activation: bool = True,
    ):
        """
        Args:
            dim_in: Number of input features.
            dim_out: Number of output features.
            w0: SIREN omega_0 factor used in weight init and Sine activation.
            is_first: Whether this is the first layer of the model.
            is_last: Whether this is the last layer of the model.
            modulate_scale: If True, apply scale modulation.
            modulate_shift: If True, apply shift modulation.
            apply_activation: If True, apply sine activation.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.apply_activation = apply_activation

        # Optional modulation and activation
        if modulate_scale or modulate_shift:
            self.modulation = FiLM(dim_in, modulate_scale, modulate_shift)

        if self.apply_activation:
            self.activation = Sine(w0)

        # Initialise linear transform. Follow weights initialization from SIREN paper.
        self.init_range = 1 / dim_in if is_first else np.sqrt(6 / dim_in) / w0

        weight = torch.zeros([dim_out, dim_in])
        weight.uniform_(-self.init_range, self.init_range)

        bias = torch.zeros([dim_out])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., dim_in] input features.

        Returns:
            [..., dim_out] output features; if is_last=True, a 0.5 shift is added.
        """
        x = F.linear(x, self.weight, self.bias)

        if self.is_last:
            # Assuming targets in [0, 1], shift by 0.5 to learn zero-centered features.
            return x + 0.5

        # Optional FiLM modulation then sine activation
        if self.modulate_scale or self.modulate_shift:
            x = self.modulation(x)
        if self.apply_activation:
            x = self.activation(x)

        return x


class LatentToModulation(nn.Module):
    """
    Maps a latent vector to FiLM modulations for each modulated layer.

    Output dict format:
      {
        layer_index: {
          "scale": [B, width] (optional),
          "shift": [B, width] (optional),
        },
        ...
      }
    """

    def __init__(
        self,
        latent_dim: int,
        layer_sizes: Optional[Tuple[int, ...]],
        width: int,
        num_modulation_layers: int,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        activation: Callable[[Tensor], Tensor] = nn.ReLU(),
    ):
        """
        Args:
            latent_dim: Dimension of the input latent vector.
            layer_sizes: Hidden layer sizes of the MLP; if None, use a single Linear.
            width: Feature width per modulated layer (matches SIREN hidden width).
            num_modulation_layers: Number of layers to modulate.
            modulate_scale: If True, produce scale modulations.
            modulate_shift: If True, produce shift modulations.
            activation: Activation function for hidden layers.
        """
        super().__init__()
        assert modulate_scale or modulate_shift, "Enable at least one of scale/shift."

        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        if layer_sizes is not None:
            self.activation = activation

        # Output size = (num_layers * width) * (#modulations per unit)
        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        self.layers = self._construct_layers()

    def _construct_layers(self) -> nn.Module:
        """Construct MLP mapping latent_dim -> output_size."""
        # Special case: No layer sizes - just create a single linear layer (recommended behaviour according to functa authors)
        if self.layer_sizes is None:
            return nn.Linear(self.latent_dim, self.output_size)

        all_sizes = (self.latent_dim,) + self.layer_sizes + (self.output_size,)
        layers = []
        for i in range(len(all_sizes) - 1):
            in_dim, out_dim = all_sizes[i], all_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            if i < len(all_sizes) - 2:
                layers.append(self.activation)

        return nn.Sequential(*layers)

    def forward(self, latent_vector: Tensor) -> Dict[int, Dict[str, Tensor]]:
        """
        Args:
            latent_vector: [B, latent_dim] latent codes.

        Returns:
            Dictionary mapping layer index to modulation dicts (scale/shift).
        """
        modulations = self.layers(latent_vector)  # [B, output_size]
        outputs: Dict[int, Dict[str, Tensor]] = {}

        for i in range(self.num_modulation_layers):
            single_layer_modulations: Dict[str, Tensor] = {}
            # Note that we add 1 to scales so that outputs of MLP will be centered
            # (since scale = 1 corresponds to identity function)
            if self.modulate_scale and self.modulate_shift:
                start = 2 * self.width * i
                single_layer_modulations["scale"] = (
                    modulations[..., start : start + self.width] + 1
                )
                single_layer_modulations["shift"] = modulations[
                    ..., start + self.width : start + 2 * self.width
                ]
            elif self.modulate_scale:
                start = self.width * i
                single_layer_modulations["scale"] = (
                    modulations[..., start : start + self.width] + 1
                )
            elif self.modulate_shift:
                start = self.width * i
                single_layer_modulations["shift"] = modulations[
                    ..., start : start + self.width
                ]
            outputs[i] = single_layer_modulations

        return outputs


class LatentModulatedSiren(nn.Module):
    """
    SIREN whose hidden layers are modulated via FiLM parameters produced from a latent.

    The latent vector is mapped to per-layer FiLM scale/shift, which are then applied
    before the sine activation in each hidden layer.
    """

    def __init__(
        self,
        width: int = 256,
        depth: int = 5,
        dim_in: int = 2,
        dim_out: int = 3,
        latent_dim: int = 64,
        layer_sizes: Optional[Tuple[int, ...]] = (256, 512),
        w0: float = 1.0,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        final_activation: Optional[Literal["sigmoid"]] = None,
        latent_init_scale: float = 0.01,
        use_meta_sgd: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            width: Hidden layer width.
            depth: Number of layers.
            dim_in: Coordinate dimension (e.g., 2 for 2D images).
            dim_out: Output channels (e.g., 3 for RGB).
            latent_dim: Dimension of latent code.
            layer_sizes: Hidden sizes for the latent-to-modulation MLP; None => single Linear.
            w0: SIREN omega_0.
            modulate_scale: Whether to learn scale modulations.
            modulate_shift: Whether to learn shift modulations.
            final_activation: Optional "sigmoid" on the final output.
            latent_init_scale: Typical scale for initializing latents (used externally).
            use_meta_sgd: If True, create learnable per-latent LR container.
        """
        super().__init__()
        self.width = width
        self.depth = depth
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.latent_init_scale = latent_init_scale
        self.use_meta_sgd = use_meta_sgd

        # Parse final activation
        if final_activation is None:
            self.final_activation: Optional[nn.Module] = None
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Incorrect final_activation string passed: {final_activation}"
            )

        if self.use_meta_sgd:
            self.meta_sgd_lrs = MetaSGDLrs(latent_dim)

        # Sine activation used after applying FiLM in hidden layers
        self.sinew0 = Sine(w0)

        # Map latent -> per-layer FiLM parameters
        self.latent_to_modulation = LatentToModulation(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth - 1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
        )

        # Backbone layers (modulation and activation are applied externally)
        self.layers = self._construct_layers()

    def _construct_layers(self) -> nn.ModuleList:
        """
        Build the SIREN layers with modulation/activation disabled internally.
        """
        layers = []
        # First layer
        layers.append(
            ModulatedSirenLayer(
                dim_in=self.dim_in,
                dim_out=self.width,
                w0=self.w0,
                is_first=True,
                modulate_scale=False,
                modulate_shift=False,
                apply_activation=False,
            )
        )

        # Hidden layers
        for _ in range(self.depth - 2):
            layers.append(
                ModulatedSirenLayer(
                    dim_in=self.width,
                    dim_out=self.width,
                    w0=self.w0,
                    modulate_scale=False,
                    modulate_shift=False,
                    apply_activation=False,
                )
            )

        # Final layer
        layers.append(
            ModulatedSirenLayer(
                dim_in=self.width,
                dim_out=self.dim_out,
                w0=self.w0,
                is_last=True,
                modulate_scale=False,
                modulate_shift=False,
                apply_activation=False,
            )
        )

        return nn.ModuleList(layers)

    def forward(self, coords: Tensor, latent: Tensor) -> Tensor:
        """
        Evaluate the latent-modulated SIREN.

        Args:
            coords: [B, N, dim_in] coordinate samples per batch.
            latent: [B, latent_dim] latent codes (one per batch item).

        Returns:
            [B, N, dim_out] predictions at coords.
        """
        B = coords.shape[0]
        # Sanity checks
        assert latent.shape[0] == B, "Batch size mismatch between coords and latent."
        assert coords.shape[-1] == self.dim_in, "Coordinate dimension mismatch."

        # 1) compute all FiLM modulations: Dict[layer_idx] -> {"scale": [B, C], "shift": [B, C]}
        mods = self.latent_to_modulation(latent)

        # 2) Forward through all but final layer, applying FiLM then Sine
        x = coords
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            m = mods[i]
            if "scale" in m:
                x = x * m["scale"].unsqueeze(1)  # [B, 1, C] -> broadcast across N
            if "shift" in m:
                x = x + m["shift"].unsqueeze(1)
            x = self.sinew0(x)

        # 3) Final linear (optionally followed by final activation)
        x = self.layers[-1](x)  # [B, N, dim_out]
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    # def reconstruct_image(self, sampling_grid, latent_vectors):
    #     B, H, W, C = sampling_grid.shape
    #     x_in = einops.rearrange(sampling_grid, "b h w c -> b (h w) c")  # HWxDim_in
    #     x_out = self.forward(x_in, latent_vectors)  # HWxDim_out
    #     x_out = torch.reshape(x_out, [B, H, W, self.dim_out])
    #     return x_out

    def reconstruct_image(
        self, sampling_grid: Tensor, latent_vectors: Tensor
    ) -> Tensor:
        """
        Reconstruct predictions for a dense coordinate grid.

        Args:
            sampling_grid: [B, *spatial, dim_in] coordinate grid.
            latent_vectors: [B, latent_dim] latent codes.

        Returns:
            [B, *spatial, dim_out] predictions reshaped to the grid.
        """
        B = sampling_grid.shape[0]
        spatial_shape = sampling_grid.shape[1:-1]
        # Flatten all spatial axes -> [B, N, dim_in]
        x_in = einops.rearrange(sampling_grid, "b ... c -> b (...) c")
        # Forward and reshape back
        x_out = self.forward(x_in, latent_vectors)  # [B, N, dim_out]
        x_out = x_out.reshape(B, *spatial_shape, self.dim_out)
        return x_out
