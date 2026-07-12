# ruff: noqa: E741
# Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas KulhanekS. All rights reserved.
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
Python package for combining 3DGS with volume rendering to enable water/fog modeling.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from pureseags._torch_impl import quat_to_rotmat
from pureseags.project_gaussians import project_gaussians
from pureseags.rasterize import rasterize_gaussians
from pureseags.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@dataclass
class PureSeaGSModelConfig(ModelConfig):
    """Water Splatting Model Config"""
    ssim_lambda: float = 0.4
    """weight of ssim loss"""
    depth_lambda: float = 0
    """weight of depth anything soft constraint"""
    _target: Type = field(default_factory=lambda: PureSeaGSModel)
    num_steps: int = 5000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.5
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    zero_medium: bool = False
    """If True, zero out the medium field"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians (0.0004, 0.0008)"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    main_loss: Literal["l1", "reg_l1", "reg_l2"] = "reg_l1"
    """main loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim"] = "reg_ssim"
    """ssim loss to use"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for medium MLP."""
    dcp_lambda: float = 0.05
    """weight of dark channel prior loss"""
    dcp_patch_size: int = 15
    """patch size for dark channel prior"""
    structure_penalty_weight: float = 0.1
    """weight of structure penalty"""
    use_global_medium: bool = False
    """If True, use global trainable constants instead of AMF MLP"""
    use_depth_guided_medium: bool = False
    """If True, concat rendered depth from previous step to direction encoding for medium MLP"""
    depth_feature_dim: int = 4
    """Dimension of depth features projected by Linear(1->depth_feature_dim) for Depth-Guided AMF"""


class PureSeaGSModel(Model):
    """
    Args:
        config: Water Splatting configuration to instantiate model
    """

    config: PureSeaGSModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # initialize the medium MLP
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        
        # medium MLP config parsing
        num_layers_medium=self.config.num_layers_medium
        hidden_dim_medium=self.config.hidden_dim_medium
        self.medium_density_bias=self.config.medium_density_bias
        
        # if type is tuple, then [0]
        num_layers_medium = num_layers_medium if isinstance(num_layers_medium, int) else num_layers_medium[0]
        hidden_dim_medium = hidden_dim_medium if isinstance(hidden_dim_medium, int) else hidden_dim_medium[0]
        self.medium_density_bias = self.medium_density_bias if isinstance(self.medium_density_bias, float) else self.medium_density_bias[0]
        
        # Depth-guided medium: cache prev step's depth for 1-step delay
        self.prev_depth_map = None

        # Depth projection layer: 1-channel depth norm -> depth_feature_dim features
        if self.config.use_depth_guided_medium:
            self.depth_proj = nn.Linear(1, self.config.depth_feature_dim)
        # Medium network input dimension (16 from SH encoding + optional depth_feature_dim)
        medium_in_dim = self.direction_encoding.get_out_dim() + (self.config.depth_feature_dim if self.config.use_depth_guided_medium else 0)
        if self.config.use_global_medium:
            # 消融实验: 使用可训练的全局常数替代 AMF MLP
            CONSOLE.log("[Ablation] Using global trainable constants instead of AMF MLP (w/o AMF)")
            self.global_medium_rgb = torch.nn.Parameter(torch.tensor([0.4, 0.5, 0.6]))
            self.global_medium_bs = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
            self.global_medium_attn = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
            # 创建占位 MLP 空参数，保持 get_param_groups 兼容
            self.medium_backbone = nn.Linear(medium_in_dim, 3)  # 最小线性层占位
            self.head_rgb = nn.Identity()
            self.head_bs = nn.Identity()
            self.head_attn = nn.Identity()
        else:
            # Medium network: Shared Backbone + Multi-Head prediction
            if num_layers_medium > 1:
                self.medium_backbone = MLP(
                    in_dim=medium_in_dim,
                    num_layers=num_layers_medium,
                    layer_width=hidden_dim_medium,
                    out_dim=hidden_dim_medium,
                    activation=nn.ReLU(),
                    out_activation=nn.ReLU(),
                    implementation=self.config.mlp_type,
                )
            else:
                self.medium_backbone = nn.Linear(medium_in_dim, hidden_dim_medium)
                self.config.mlp_type = "torch"

            # 2. 独立的物理属性预测头 (Multi-Heads)
            self.head_rgb = nn.Linear(hidden_dim_medium, 3)
            self.head_bs = nn.Linear(hidden_dim_medium, 3)
            self.head_attn = nn.Linear(hidden_dim_medium, 3)

        # ------------------------Gaussians Initialization------------------------
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157]) 
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    


    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
    
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                # reset the exp of optimizer
                for key in ["medium_mlp", "direction_encoding"]:
                    optim = optimizers.optimizers[key]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                
            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:                
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers"""
        gps = self.get_gaussian_param_groups()
        
        if self.config.use_global_medium:
            # 消融实验：全局常数模式，将全局参数放入 optimizer
            gps["medium_mlp"] = [self.global_medium_rgb, self.global_medium_bs, self.global_medium_attn]
        else:
            # 将主干网络和所有的预测头打包进同一个 param group
            gps["medium_mlp"] = (
                list(self.medium_backbone.parameters()) +
                list(self.head_rgb.parameters()) +
                list(self.head_bs.parameters()) +
                list(self.head_attn.parameters())
            )
        
        # depth_proj 合并到 medium_mlp 共享优化器（避免单独配置 optimizer）
        if self.config.use_depth_guided_medium:
            gps["medium_mlp"] += list(self.depth_proj.parameters())
        
        gps["direction_encoding"] = list(self.direction_encoding.parameters())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # Medium
        # Encode directions
        y = torch.linspace(0., H, H, device=self.device)
        x = torch.linspace(0., W, W, device=self.device)
        yy, xx = torch.meshgrid(y, x)
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ R.T

        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)
        outputs_shape = directions.shape[:-1]

        # --- Depth-Guided Medium: concat cached depth from prev step ---
        if self.config.use_depth_guided_medium:
            target_H, target_W = outputs_shape
            if self.prev_depth_map is not None:
                # Second+ step: use cached depth from previous iteration
                prev_depth = self.prev_depth_map  # (prev_H, prev_W, 1)
                if prev_depth.shape[:2] != (target_H, target_W):
                    prev_depth = F.interpolate(
                        prev_depth.permute(2, 0, 1).unsqueeze(0),
                        size=(target_H, target_W),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                depth_norm = prev_depth / (prev_depth.max() + 1e-6)
                depth_feat = self.depth_proj(depth_norm)  # (H, W, 1) -> (H, W, depth_feature_dim)
            else:
                # First step: use zeros (MLP always expects 20D input)
                depth_feat = torch.zeros(target_H, target_W, self.config.depth_feature_dim, device=directions_flat.device)
            depth_feat_flat = depth_feat.reshape(-1, self.config.depth_feature_dim).detach()
            directions_encoded = torch.cat([directions_encoded, depth_feat_flat], dim=-1)

        # ---------------- 修改开始：多头前向传播 / 全局常量模式 ----------------
        if self.config.use_global_medium:
            # 消融实验: 使用全局可训练常数，广播到每个像素
            medium_rgb = torch.sigmoid(self.global_medium_rgb).view(1, 1, 3).expand(*outputs_shape, -1).to(directions)
            medium_bs = F.softplus(self.global_medium_bs + self.medium_density_bias).view(1, 1, 3).expand(*outputs_shape, -1).to(directions)
            medium_attn = F.softplus(self.global_medium_attn + self.medium_density_bias).view(1, 1, 3).expand(*outputs_shape, -1).to(directions)
        else:
            # 1. 提取共享物理特征
            if self.config.mlp_type == "tcnn":
                shared_features = self.medium_backbone(directions_encoded)
            else:
                shared_features = self.medium_backbone(directions_encoded.float())

            shared_features = shared_features.float()
            
            # 2. 独立预测不同物理量
            raw_rgb = self.head_rgb(shared_features)
            raw_bs = self.head_bs(shared_features)
            raw_attn = self.head_attn(shared_features)

            # 3. 激活并恢复形状
            medium_rgb = (
                self.colour_activation(raw_rgb)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            medium_bs = (
                self.sigma_activation(raw_bs + self.medium_density_bias)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            medium_attn = (
                self.sigma_activation(raw_attn + self.medium_density_bias)
                .view(*outputs_shape, -1)
                .to(directions)
            )
        # ---------------- 修改结束 ----------------

        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                        "rgb_object": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()
        
        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                    "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb), "rgb_clear_clamp": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}

        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        self.xys_grad_abs = torch.zeros_like(self.xys)

        rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            self.xys_grad_abs,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore
        
        rgb = rgb_object + rgb_medium
        rgb_clear_clamp = torch.clamp(rgb_clear, 0., 1.)
        rgb_clear = rgb_clear / (rgb_clear + 1.0)
        
        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())  

        # Cache depth for next step's Depth-Guided AMF (1-step delay)
        if self.training and self.config.use_depth_guided_medium:
            self.prev_depth_map = depth_im.detach()
                 
        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": medium_rgb, 
                "rgb_object": rgb_object, "rgb_clear": rgb_clear, "rgb_clear_clamp": rgb_clear_clamp, "rgb_medium": rgb_medium, "pred_image": rgb,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}  # type: ignore
        
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return image[..., :3]
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """计算并返回损失函数字典。
        
        优化说明：
        1. 移除 lstsq，改用数学闭式解进行视差对齐，大幅提升 FPS。
        2. 移除调试文件写入逻辑，消除 CPU-GPU 同步阻塞。
        3. 优化了内存张量的复用。
        4. 引入了高性能张量化暗通道先验（DCP）计算逻辑。
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]

        # 处理 Mask
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        else:
            mask = None

        # ==========================================================
        # 1. 基础重建 Loss (Reconstruction Loss)
        # ==========================================================
        if self.config.main_loss == "l1":
            recon_loss = torch.abs(gt_img - pred_img).mean()
        elif self.config.main_loss == "reg_l1":
            recon_loss = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
        else:
            recon_loss = (((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2).mean()
        
        # ==========================================================
        # 2. 结构一致性与背景相似度惩罚
        # ==========================================================
        penalty_weight = self.config.structure_penalty_weight
        
        if self.config.ssim_lambda > 0.0:
            if self.config.ssim_loss != "ssim":
                simloss = 1 - self.ssim((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...], 
                                        (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
            else:
                simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
                
            local_error = torch.abs(gt_img - pred_img).mean(dim=-1, keepdim=True)
            color_diff = torch.abs(pred_img - outputs["medium_rgb"]).mean(dim=-1, keepdim=True)
            bg_similarity = torch.exp(-10.0 * color_diff) 
            
            # 惩罚项计算
            penalty_map = local_error.detach() * bg_similarity.detach() * outputs["accumulation"]
            accumulation_penalty = penalty_map.mean()
            structure_penalty = simloss.detach() * accumulation_penalty

            # === 提取特征图并保存到本地 ===
            # 建议选一个固定的 step 保存，比如第 5000 步或者最后一步
            if self.step == 5000 or self.step == self.config.num_steps - 1:
                os.makedirs("paper_figures", exist_ok=True)
                # penalty_map 的 shape 是 (H, W, 1)，需要去掉单维度并转到 CPU
                penalty_vis = penalty_map.squeeze().detach().cpu().numpy()
                
                # 归一化到 0-1 方便可视化 (可选，看实际数值范围)
                # penalty_vis = (penalty_vis - penalty_vis.min()) / (penalty_vis.max() - penalty_vis.min() + 1e-8)
                
                # 使用 magma 或 inferno 色带（黑/紫到黄/白），非常适合表示“惩罚力度”
                plt.imsave(f"paper_figures/penalty_map_step_{self.step}.png", penalty_vis, cmap='magma')
        else:
            simloss = torch.tensor(0.0, device=self.device)
            structure_penalty = torch.tensor(0.0, device=self.device)

        # ==========================================================
        # 3. Depth Anything 视差对齐损失 (优化后的闭式解)
        # ==========================================================
        depth_loss = torch.tensor(0.0, device=self.device)
        depth_lambda = getattr(self.config, "depth_lambda", 0.0)
        
        if "depth_image" in batch and depth_lambda > 0.0:
            gt_disp = self._downscale_if_required(batch["depth_image"]).to(self.device).float()
            pred_depth = outputs["depth"]
            pred_disp = 1.0 / (pred_depth + 1e-6)
            
            if mask is not None:
                gt_disp = gt_disp * mask
                valid_mask = (mask > 0.5).view(-1)
            else:
                valid_mask = None

            # 拉平张量准备对齐
            pred_flat = pred_disp.view(-1)
            gt_flat = gt_disp.view(-1)
            
            if valid_mask is not None:
                pred_flat = pred_flat[valid_mask]
                gt_flat = gt_flat[valid_mask]

            # --- 闭式解线性回归 (求解 scale 和 shift) ---
            m_pred = pred_flat.mean()
            m_gt = gt_flat.mean()
            
            diff_pred = pred_flat - m_pred
            diff_gt = gt_flat - m_gt
            
            var_pred = torch.mean(diff_pred ** 2) + 1e-6
            cov_pred_gt = torch.mean(diff_pred * diff_gt)
            
            scale = cov_pred_gt / var_pred
            shift = m_gt - scale * m_pred
            # -------------------------------------------

            pred_aligned = pred_disp * scale + shift

            # === 提取对齐后的深度图 ===
            if self.step == 5000 or self.step == self.config.num_steps - 1:
                os.makedirs("paper_figures", exist_ok=True)
                # 恢复成 (H, W) 的形状
                depth_vis = pred_aligned.view(gt_img.shape[0], gt_img.shape[1]).detach().cpu().numpy()
                plt.imsave(f"paper_figures/depth_aligned_step_{self.step}.png", depth_vis, cmap='viridis')
            
            if mask is not None:
                depth_loss = torch.abs((pred_aligned - gt_disp) * mask).mean()
            else:
                depth_loss = torch.abs(pred_aligned - gt_disp).mean()

        # ==========================================================
        # 4. 暗通道先验损失 (Dark Channel Prior Loss)
        # ==========================================================
        dcp_loss = torch.tensor(0.0, device=self.device)
        dcp_lambda = getattr(self.config, "dcp_lambda", 0.0)

        if dcp_lambda > 0.0:
            # 将 (H, W, 3) 转换为 (1, 3, H, W) 以适配池化操作
            img_chw = pred_img.permute(2, 0, 1).unsqueeze(0)
            
            # --- 通道级最小化 ---
            # 选项 A：标准 DCP (适用于雾霾等常规低能见度场景)
            min_channel, _ = torch.min(img_chw, dim=1, keepdim=True)
            
            # 选项 B：水下 UDCP (排除衰减极快的 Red 通道，仅在 Green 和 Blue 中求最小)
            # 如果是用于物理光照解耦的水下重建，取消下面这行的注释并替换掉上面的选项 A
            # min_channel, _ = torch.min(img_chw[:, 1:3, :, :], dim=1, keepdim=True)
            
            # --- 空间块最小化 (利用负值最大池化实现极速 2D 最小值滤波) ---
            patch_size = getattr(self.config, "dcp_patch_size", 15)
            # stride=1 且 padding=patch_size//2 确保输出空间分辨率不变
            dark_channel = -F.max_pool2d(-min_channel, kernel_size=patch_size, stride=1, padding=patch_size//2)

            # === 提取暗通道特征图并保存到本地 ===
            if self.step == 5000 or self.step == self.config.num_steps - 1:
                os.makedirs("paper_figures", exist_ok=True)
                # dark_channel 的 shape 是 (1, 1, H, W)，需要 squeeze 成 (H, W)
                dcp_vis = dark_channel.squeeze().detach().cpu().numpy()
                
                # 使用 turbo 或 jet 色带（经典的彩虹色），常用于展示深度或物理先验
                plt.imsave(f"paper_figures/dcp_map_step_{self.step}.png", dcp_vis, cmap='turbo')
            
            # 先验假设：清晰图像的暗通道强度趋近于 0
            if mask is not None:
                # 适配 mask 维度: (H, W, 1) -> (1, 1, H, W)
                mask_chw = mask.permute(2, 0, 1).unsqueeze(0) if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0)
                dcp_loss = (dark_channel * mask_chw).mean()
            else:
                dcp_loss = dark_channel.mean()

        # ==========================================================
        # 5. 汇总 Total Loss
        # ==========================================================
        total_loss = (1 - self.config.ssim_lambda) * recon_loss + \
                     self.config.ssim_lambda * simloss + \
                     penalty_weight * structure_penalty + \
                     depth_lambda * depth_loss + \
                     dcp_lambda * dcp_loss

        return {
            "main_loss": total_loss,
            "structure_penalty": structure_penalty,
            "depth_loss": depth_loss,
            "dcp_loss": dcp_loss,
        }
    
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = predicted_rgb

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"gt": output_gt_rgb, "rgb_medium": outputs["rgb_medium"], "rgb_object": outputs["rgb_object"], "depth": outputs["depth"], "rgb": outputs["rgb"], "rgb_clear": outputs["rgb_clear"]}
        return metrics_dict, images_dict