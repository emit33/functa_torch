from ctypes import Array
from typing import Callable, Dict, Literal, Optional, Tuple
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class Sine(nn.Module):
    """Applies a scaled sine transform to input: out = sin(w0 * in)."""

    def __init__(self, w0: float = 1.0):
        """
        Args:
          w0 (float): Scale factor in sine activation (omega_0 factor from SIREN).
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


class FiLM(nn.Module):
    """Applies a FiLM modulation: out = scale * in + shift.

    Notes:
      We currently initialize FiLM layers as the identity. However, this may not
      be optimal. In pi-GAN for example they initialize the layer with a random
      normal.
    """

    def __init__(self, dim_in, modulate_scale=True, modulate_shift=True):
        """Constructor.

        Args:
          modulate_shift (bool): Whether to apply a shift modulation.
          modulate_scale (bool): Whether to apply a scale modulation.
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
    """A modulated SIREN layer.

    This layer applies a Sine activation to the input, and modulates it with
    FiLM parameters.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        is_first=False,
        is_last=False,
        modulate_shift: bool = True,
        modulate_scale: bool = True,
        apply_activation: bool = True,
    ):
        """Constructor.

        Args:
        f_in (int): Number of input features.
        f_out (int): Number of output features.
        w0 (float): Scale factor in sine activation.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        apply_activation: If True, applies sine activation.
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

        # Define option modulation and activations
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
        x = F.linear(x, self.weight, self.bias)

        if self.is_last:
            # We assume target data (e.g. RGB values of pixels) lies in [0, 1]. To
            # learn zero-centered features we therefore shift output by .5
            return x + 0.5

        # Optionally apply modulation
        if self.modulate_scale or self.modulate_shift:
            x = self.modulation(x)

        # Optionally apply activation
        if self.apply_activation:
            x = self.activation(x)

        return x


class ModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations as in pi-GAN."""

    def __init__(
        self,
        width: int = 256,
        depth: int = 5,
        dim_in: int = 3,
        dim_out: int = 3,
        w0: float = 1.0,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        use_meta_sgd: bool = False,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        if use_meta_sgd:
            raise NotImplementedError("Meta SGD not yet implemented")

        if depth < 3:
            raise ValueError("Depth must be >=3 to permit at least one hidden layer.")

        self.layers = self._construct_layers()

    def _construct_layers(self):
        layers = []

        # First layer
        layers.append(
            ModulatedSirenLayer(
                dim_in=self.dim_in,
                dim_out=self.width,
                w0=self.w0,
                is_first=True,
                modulate_scale=self.modulate_scale,
                modulate_shift=self.modulate_shift,
            )
        )

        # Hidden layers
        for _ in range(self.depth - 2):
            layers.append(
                ModulatedSirenLayer(
                    dim_in=self.width,
                    dim_out=self.width,
                    w0=self.w0,
                    modulate_scale=self.modulate_scale,
                    modulate_shift=self.modulate_shift,
                )
            )

        # Final layer
        layers.append(
            ModulatedSirenLayer(
                dim_in=self.width,
                dim_out=self.dim_out,
                w0=self.w0,
                is_last=True,
                modulate_scale=self.modulate_scale,
                modulate_shift=self.modulate_shift,
            )
        )

        return nn.Sequential(*layers)

    def forward(self, coords):
        """Evaluates model at a batch of coordinates.

        Args:
        coords (Array): Array of coordinates. Should have shape (height, width, 2)
            for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
        Output features at coords.
        """
        # Flatten coordinates
        x = torch.reshape(coords, (-1, coords.shape[-1]))

        out = self.layers(x)

        return torch.reshape(out, list(coords.shape[:-1]) + [self.dim_out])


class LatentToModulation(nn.Module):
    """Function mapping latent vector to a set of modulations."""

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
        """Constructor.

        Args:
          latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
          layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
          width: Width of each hidden layer in MLP of function rep.
          num_modulation_layers: Number of layers in MLP that contain modulations.
          modulate_scale: If True, returns scale modulations.
          modulate_shift: If True, returns shift modulations.
          activation: Activation function to use in MLP.
        """
        super().__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift

        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes  # counteract XM that converts to list
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        if not layer_sizes is None:
            self.activation = activation

        # MLP outputs all modulations. We apply modulations on every hidden unit
        # (i.e on width number of units) at every modulation layer.
        # At each of these we apply either a scale or a shift or both,
        # hence total output size is given by following formula
        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        self.layers = self._construct_layers()

    def _construct_layers(self):
        """Construct MLP layers from prescribed widths."""
        # Handle case where layer_sizes is none: ie just have a linear layer
        if self.layer_sizes is None:
            return nn.Linear(self.latent_dim, self.output_size)

        # Otherwise construct proper MLP
        all_sizes = (self.latent_dim,) + self.layer_sizes + (self.output_size,)

        layers = []

        # Create layers by pairing consecutive sizes
        for i in range(len(all_sizes) - 1):
            in_dim = all_sizes[i]
            out_dim = all_sizes[i + 1]

            # Add linear layer
            layers.append(nn.Linear(in_dim, out_dim))

            # Add activation (except for the final layer)
            if i < len(all_sizes) - 2:
                layers.append(self.activation)

        return nn.Sequential(*layers)

    def forward(self, latent_vector: Tensor) -> Dict[int, Dict[str, Tensor]]:
        modulations = self.layers(latent_vector)

        outputs = {}
        # Partition modulations into scales and shifts at every layer
        for i in range(self.num_modulation_layers):
            single_layer_modulations = {}
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
        device: Optional[torch.device] = None,
    ):
        """Constructor.

        Args:
          width (int): Width of each hidden layer in MLP.
          depth (int): Number of layers in MLP.
          dim_in (int): Number of input channels
          dim_out (int): Number of output channels.
          latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
          layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
          w0 (float): Scale factor in sine activation in first layer.
          modulate_scale: If True, modulates scales.
          modulate_shift: If True, modulates shifts.
          latent_init_scale: Scale at which to randomly initialize latent vector.
          use_meta_sgd: Whether to use meta-SGD.
          meta_sgd_init_range: Range from which initial meta_sgd learning rates will
            be uniformly sampled.
          meta_sgd_clip_range: Range at which to clip learning rates.
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
            self.final_activation = None
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Incorrect final_activation string passed: {final_activation}"
            )

        if self.use_meta_sgd:
            raise NotImplementedError("Meta SGD not yet implemented")

        # Initialise Sine activations
        self.sinew0 = Sine(w0)

        self.latent_to_modulation = LatentToModulation(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth - 1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
        )

        # Obtain layers
        self.layers = self._construct_layers()

    def _construct_layers(self):
        layers = []

        # Note all modulations are set to False here, since we apply modulations
        # from latent_to_modulations output. Similarly, activations are set to false
        # so that modulations can be applied prior to activation
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

    # def modulate(self, x: Tensor, modulations: Dict[str, Tensor]) -> Tensor:
    #     """Modulates input according to modulations.

    #     Args:
    #     x: Hidden features of MLP.
    #     modulations: Dict with keys 'scale' and 'shift' (or only one of them)
    #         containing modulations.

    #     Returns:
    #     Modulated vector.
    #     """
    #     if "scale" in modulations:
    #         x = modulations["scale"] * x
    #     if "shift" in modulations:
    #         x = x + modulations["shift"]
    #     return x

    # def forward(self, coords: Tensor, latent_vector) -> Tensor:
    #     """Evaluates model at a batch of coordinates.

    #     Args:
    #     coords (Array): Array of coordinates. Should have shape (height, width, 2)
    #         for images and (depth/time, height, width, 3) for 3D shapes/videos.

    #     Returns:
    #     Output features at coords.
    #     """
    #     # Check coordinate dimensions
    #     assert (
    #         coords.shape[-1] == self.dim_in
    #     ), f"Expected {self.dim_in} coordinate dimensions, got {coords.shape[-1]}"

    #     # Compute modulations from latent vector
    #     modulations = self.latent_to_modulation(latent_vector)

    #     # Flatten coordinates
    #     x = torch.reshape(coords, (-1, coords.shape[-1]))

    #     # Layers before final layer
    #     for i, layer in enumerate(self.layers[:-1]):
    #         x = layer(x)
    #         x = self.modulate(x, modulations[i])
    #         x = self.sinew0(x)

    #     out = self.layers[-1](x)

    #     if self.final_activation is not None:
    #         out = self.final_activation(out)

    #     return torch.reshape(out, list(coords.shape[:-1]) + [self.dim_out])

    def forward(
        self, coords: Tensor, latent: Tensor  # [B, N_points, dim_in]  # [B, latent_dim]
    ) -> Tensor:
        B = coords.shape[0]

        # Check correct dimensionality
        assert latent.shape[0] == B
        assert coords.shape[-1] == self.dim_in

        # 1) compute all FiLM modulations: Dict[layer_idx] -> {"scale": [B, C], "shift": [B, C]}
        mods = self.latent_to_modulation(latent)

        # Pass through
        x = coords
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            m = mods[i]
            if "scale" in m:
                x = x * m["scale"].unsqueeze(1)  # [B,1,C] -> broadcast over N
            if "shift" in m:
                x = x + m["shift"].unsqueeze(1)

            # sine activation
            x = self.sinew0(x)

        # 4) final linear (+ optional activation)
        x = self.layers[-1](x)  # [B, N, dim_out]
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    # def reconstruct_image(self, sampling_grid, latent_vector):

    #     x_in = einops.rearrange(sampling_grid, "h w c -> (h w) c")  # HWxDim_in
    #     x_out = self.forward(x_in, latent_vector)  # HWxDim_out
    #     x_out = torch.reshape(x_out, list(sampling_grid.shape[:-1]) + [self.dim_out])
    #     return x_out

    def reconstruct_image(self, sampling_grid, latent_vectors):
        B, H, W, C = sampling_grid.shape
        x_in = einops.rearrange(sampling_grid, "b h w c -> b (h w) c")  # HWxDim_in
        x_out = self.forward(x_in, latent_vectors)  # HWxDim_out
        x_out = torch.reshape(x_out, [B, H, W, self.dim_out])
        return x_out
