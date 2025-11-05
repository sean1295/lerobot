#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Equilibrium Matching Policy as per "Equilibrium Matching: A Unified Framework for Conditional Generation"

TODO(alexander-soare):
 - Remove reliance on diffusers for LR scheduler.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from lerobot.policies.equilibrium_matching.configuration_equilibrium_matching import EquilibriumMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, LANG_INSTRUCTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class EquilibriumMatchingPolicy(PreTrainedPolicy):
    """
    Equilibrium Matching Policy, where the vector field is defined as the gradient of a learned
    energy-based model (EBM).
    (paper: https://huggingface.co/papers/2405.15548).
    """

    config_class = EquilibriumMatchingConfig
    name = "equilibrium_matching"

    def __init__(
        self,
        config: EquilibriumMatchingConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.eqm = EquilibriumMatchingModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.eqm.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.language_conditioned:
            self._queues[LANG_INSTRUCTION] = deque(maxlen=1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1) if k != LANG_INSTRUCTION else self._queues[k][0]
            for k in batch
            if k in self._queues
        }
        actions = self.eqm.generate_actions(batch)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        loss_dict = dict()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [
                    batch[key].unsqueeze(1) if batch[key].ndim == 4 else batch[key]
                    for key in self.config.image_features
                ],
                dim=-4,
            )

        loss = self.eqm.compute_loss(batch)
        loss_dict["eqm_loss"] = loss.item()

        return loss, loss_dict


class EquilibriumMatchingModel(nn.Module):
    def __init__(self, config: EquilibriumMatchingConfig):
        super().__init__()
        self.config = config

        # Build observation encoders.
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [RgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = RgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
        if self.config.language_conditioned:
            self.lang_encoder = LanguageEncoder(config)
            self.lang_encoder.requires_grad_(False)
            global_cond_dim += self.lang_encoder.feature_dim

        self.unet = ConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

    def _get_vector_field(self, x_t: Tensor, time: Tensor, global_cond: Tensor, create_graph: bool = False) -> Tensor:
        """
        Computes the vector field v_t by taking the gradient of a learned energy function.
        """
        # Input x_t needs to track gradients for the autograd call
        x_t.requires_grad_(True)

        # 1. Get the base output from the policy network.
        v_pred = self.unet(x_t, time, global_cond)

        # 2. Define the scalar energy function E(x_t).
        if self.config.ebm_type == 'dot':
            # Energy is the dot product between the network output and the input state.
            energy = torch.sum(v_pred * x_t, dim=(-1, -2))
        elif self.config.ebm_type == 'l2':
            # Energy is the squared L2 norm of the network output.
            energy = -torch.sum(v_pred**2, dim=(-1, -2)) / 2
        else:
            raise ValueError(f"Unknown EBM type: {self.config.ebm_type}")

        # 3. Compute the final vector field v_t = ∇_x_t E(x_t).
        v_t = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_t,
            create_graph=create_graph
        )[0]

        return v_t

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Computes the Equilibrium Matching loss.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        
        action_batch = batch[ACTION]
        global_cond = self._prepare_global_conditioning(batch)

        # Sample time from a Beta distribution for the flow matching path
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((action_batch.shape[0],)).to(device=action_batch.device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001 # Clamp time to (0.001, 1.0)

        # Generate noise and construct the interpolated state x_t
        noise = torch.randn_like(action_batch, device=action_batch.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action_batch

        # Define the target vector field for the straight path
        u_t = noise - action_batch

        # Get the model's predicted vector field using the EqM method
        v_t = self._get_vector_field(
            x_t, time, global_cond,
            create_graph=True  # Create graph for backpropagation
        )

        loss = F.mse_loss(u_t, v_t, reduction="none")

        # Mask loss wherever the action is padded.
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        
        return loss.mean()

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Generates actions by solving the ODE from t=1 to t=0.
        """
        batch_size = batch["observation.state"].shape[0]
        with torch.enable_grad():
            global_cond = self._prepare_global_conditioning(batch)
            
        device = get_device_from_parameters(self)

        # Start with pure noise
        x_t = torch.randn(
            (batch_size, self.config.horizon, self.config.action_feature.shape[0]), device=device
        )

        # Define the time step for the ODE solver
        dt = -1.0 / self.config.num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Solve the ODE from t=1 to t=0 using the Euler method
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            
            # Re-enable gradients locally for the EqM calculation
            with torch.enable_grad():
                v_t = self._get_vector_field(
                    x_t, expanded_time, global_cond,
                    create_graph=False # No graph needed for inference
                )

            # Perform the Euler step
            x_t = x_t + dt_tensor * v_t
            time += dt_tensor

        # Extract `n_action_steps` worth of actions.
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = x_t[:, start:end]

        return actions

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        if self.config.language_conditioned:
            assert LANG_INSTRUCTION in batch
            lang_emb = self.lang_encoder(batch[LANG_INSTRUCTION])
            global_cond_feats.append(lang_emb.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1))

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113).
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class RgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector."""

    def __init__(self, config: EquilibriumMatchingConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2 if config.spatial_softmax_num_keypoints else 1024

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return x


class LanguageEncoder(nn.Module):
    """Encodes a language instruction into a 1D feature vector."""

    def __init__(self, config: EquilibriumMatchingConfig):
        super().__init__()
        tokenizer = config.tokenizer
        self.lang_emb_model = AutoModel.from_pretrained(tokenizer, torch_dtype=torch.float16).eval()
        self.tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)
        self.tokenizer_max_length = config.tokenizer_max_length
        self.feature_dim = self.forward("DUMMY INPUT").shape[-1]

    def forward(self, lang):
        device = get_device_from_parameters(self.lang_emb_model)
        if isinstance(lang, str):
            lang = [lang]
        tokens = self.tz(
            text=lang,
            add_special_tokens=True,
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        # lang_emb = self.lang_emb_model(**tokens).last_hidden_state[:, -1]
        lang_emb = self.lang_emb_model.get_text_features(**tokens).detach()
        return lang_emb


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d --> SwapAxes --> RMSNorm --> SwapAxes --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.RMSNorm(out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = x.swapaxes(-1, -2)
        x = self.norm(x)
        x = x.swapaxes(-1, -2)
        return self.act(x)


class ConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""

    def __init__(self, config: EquilibriumMatchingConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                ConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, time: Tensor, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        time_embed = self.time_encoder(time)

        if global_cond is not None:
            global_feature = torch.cat([time_embed, global_cond], axis=-1)
        else:
            global_feature = time_embed

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class ConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
