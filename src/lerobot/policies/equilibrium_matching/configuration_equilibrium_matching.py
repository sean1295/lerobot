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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, SOAPConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("equilibrium_matching")
@dataclass
class EquilibriumMatchingConfig(PreTrainedConfig):
    """Configuration class for EquilibriumMatchingPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Action prediction horizon size.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary specifying the normalization mode for input features.
        output_normalization_modes: A dictionary specifying the normalization mode for output features.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone.
        crop_is_random: Whether the crop should be random at training time.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoders_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the Unet.
        kernel_size: The convolutional kernel size of the Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The embedding dimension for the time conditioning.
        use_film_scale_modulation: Whether to use scale modulation in FiLM conditioning.
        num_steps: Number of ODE solver steps for inference.
        ebm_type: Type of energy function for the EBM. Can be 'dot' or 'l2'.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions.
    """

    # Inputs / output structure.
    n_obs_steps: int = 1
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.QUANTILES,  
            "ACTION": NormalizationMode.QUANTILES,  
        }
    )

    drop_n_last_frames: int = 0  # horizon - n_action_steps - n_obs_steps + 1

    use_bc: bool = True
    # Whether to use CLASS objective (in-placement modification)
    use_class: bool = False
    # How many actions to precompute for CLASS (in-placement modification)
    class_num_actions_to_store: int = 10000
    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (224, 224)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 0
    use_separate_rgb_encoder_per_camera: bool = False
    # Language conditioning.
    language_conditioned: bool = False
    tokenizer: str = "openai/clip-vit-base-patch32"
    tokenizer_max_length: int = 72
    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Equilibrium Matching specific parameters
    num_steps: int = 16
    ebm_type: str = "dot"  # 'dot' or 'l2'

    # Loss computation
    do_mask_loss_for_padding: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_ebm_types = ["dot", "l2"]
        if self.ebm_type not in supported_ebm_types:
            raise ValueError(
                f"`ebm_type` must be one of {supported_ebm_types}. Got {self.ebm_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_optimizer_preset(self) -> SOAPConfig:
        return SOAPConfig(
            lr=1e-3,
            betas=(0.95, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
