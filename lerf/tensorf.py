# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
TensorRF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, cast

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    TensorCPEncoding,
    TensorVMEncoding,
    TriplaneEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from lerf.tensorf_field import TensoRFField
from nerfstudio.model_components.losses import (
    MSELoss,
    tv_loss,
    distortion_loss,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class TensoRFModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: TensoRFModel)
    """target class to instantiate"""
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 640
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "tv_reg_density": 1e-3,
            "tv_reg_color": 1e-4,
            "l1_reg": 5e-4,
        }
    )
    """Loss specific weights."""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""
    tensorf_encoding: Literal["triplane", "vm", "cp"] = "vm"
    regularization: Literal["none", "l1", "tv"] = "tv"
    """Regularization method used in tensorf paper"""
    # Proposal Sampler Config
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    num_proposal_iterations: int = 2
    """Scales n from 1 to proposal_update_every over this many steps"""
    use_same_proposal_network: bool = False
    """Number of proposal network iterations."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    predict_normals = False


class TensoRFModel(Model):
    """TensoRF Model

    Args:
        config: TensoRF configuration to instantiate model
    """

    config: TensoRFModelConfig

    def __init__(
        self,
        config: TensoRFModelConfig,
        **kwargs,
    ) -> None:
        print(f"final resolution is {config.final_resolution}")
        self.init_resolution = config.init_resolution
        self.upsampling_iters = config.upsampling_iters
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
        self.upsampling_steps = (
            np.round(
                np.exp(
                    np.linspace(
                        np.log(config.init_resolution),
                        np.log(config.final_resolution),
                        len(config.upsampling_iters) + 1,
                    )
                )
            )
            .astype("int")
            .tolist()[1:]
        )
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # the callback that we want to run every X iterations after the training iteration
        def reinitialize_optimizer(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
            assert training_callback_attributes.optimizers is not None
            assert training_callback_attributes.pipeline is not None
            index = self.upsampling_iters.index(step)
            resolution = self.upsampling_steps[index]

            # upsample the position and direction grids
            self.field.density_encoding.upsample_grid(resolution)
            self.field.color_encoding.upsample_grid(resolution)

            # reinitialize the encodings optimizer
            optimizers_config = training_callback_attributes.optimizers.config
            enc = training_callback_attributes.pipeline.get_param_groups()["encodings"]
            lr_init = optimizers_config["encodings"]["optimizer"].lr

            training_callback_attributes.optimizers.optimizers["encodings"] = optimizers_config["encodings"][
                "optimizer"
            ].setup(params=enc)
            if optimizers_config["encodings"]["scheduler"]:
                training_callback_attributes.optimizers.schedulers["encodings"] = (
                    optimizers_config["encodings"]["scheduler"]
                    .setup()
                    .get_scheduler(
                        optimizer=training_callback_attributes.optimizers.optimizers["encodings"], lr_init=lr_init
                    )
                )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.upsampling_iters,
                func=reinitialize_optimizer,
                args=[self, training_callback_attributes],
            )
        ]
        return callbacks

    def update_to_step(self, step: int) -> None:
        if step < self.upsampling_iters[0]:
            return

        new_iters = list(self.upsampling_iters) + [step + 1]
        new_iters.sort()

        index = new_iters.index(step + 1)
        new_grid_resolution = self.upsampling_steps[index - 1]

        self.field.density_encoding.upsample_grid(new_grid_resolution)  # type: ignore
        self.field.color_encoding.upsample_grid(new_grid_resolution)  # type: ignore

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        if self.config.tensorf_encoding == "vm":
            density_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "cp":
            density_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "triplane":
            density_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        else:
            raise ValueError(f"Encoding {self.config.tensorf_encoding} not supported")

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = TensoRFField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            use_sh=False,
        )


        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color="random")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # regularizations
        if self.config.tensorf_encoding == "cp" and self.config.regularization == "tv":
            raise RuntimeError("TV reg not supported for CP decomposition")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())

        return param_groups

    def get_rgb_depth_acc(self, ray_samples: RaySamples):
        # ray_samples_list.append(ray_samples)
        # uniform sampling
        # ray_samples_uniform = self.sampler_uniform(ray_bundle)
        # dens = self.field.get_density(ray_samples)
        # weights = ray_samples.get_weights(dens)
        # weights_list.append(weights)
        # coarse_accumulation = self.renderer_accumulation(weights)
        # acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        # ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples, weights)
        # ray_samples_list.append(ray_samples_pdf)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples, mask=None, bg_color=None
        )

        weights_fine = ray_samples.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        # weights_list.append(weights_fine)

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)
        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs, weights_fine

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}

        if self.config.regularization == "l1":
            l1_parameters = []
            for parameter in self.field.density_encoding.parameters():
                l1_parameters.append(parameter.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()
        elif self.config.regularization == "tv":
            density_plane_coef = self.field.density_encoding.plane_coef
            color_plane_coef = self.field.color_encoding.plane_coef
            assert isinstance(color_plane_coef, torch.Tensor) and isinstance(
                density_plane_coef, torch.Tensor
            ), "TV reg only supported for TensoRF encoding types with plane_coef attribute"
            loss_dict["tv_reg_density"] = tv_loss(density_plane_coef)
            loss_dict["tv_reg_color"] = tv_loss(color_plane_coef)
        elif self.config.regularization == "none":
            pass
        else:
            raise ValueError(f"Regularization {self.config.regularization} not supported")

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            # near_plane=self.config.collider_params["near_plane"],
            # far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = cast(torch.Tensor, self.ssim(image, rgb))
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i
        return metrics_dict, images_dict