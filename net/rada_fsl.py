"""RADA-FSL episodic model wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from net.fewshot_common import BaseConv64FewShotModel, pooled_episode_features
from net.heads.rada_head import RADAFewShotHead


class RADAFSL(nn.Module):
    """Wrapper that reuses the existing encoder and applies the RADA head."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        fsl_mamba_base_dim: int = 48,
        fsl_mamba_output_dim: int = 320,
        fsl_mamba_drop_path: float = 0.02,
        fsl_mamba_perturb_sigma: float = 0.0,
        tau_r: float = 0.5,
        lambda_proto: float = 0.7,
        gamma_disp: float = 0.5,
        eps: float = 1e-6,
        reliability_head: str = "linear",
        reliability_hidden_dim: int | None = None,
        l2_normalize: bool = True,
        entropy_reg_weight: float = 0.0,
        use_reliability: bool = True,
        use_dispersion_metric: bool = True,
        query_conditioned: bool = True,
        use_residual_anchor: bool = True,
        use_shrinkage: bool = True,
        disp_clamp_max: float | None = None,
        use_evidence_bound: bool = True,
        evidence_temperature: float = 1.0,
        min_reliability_mix: float = 0.25,
        dispersion_inflation: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone_model = BaseConv64FewShotModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
            fsl_mamba_base_dim=fsl_mamba_base_dim,
            fsl_mamba_output_dim=fsl_mamba_output_dim,
            fsl_mamba_drop_path=fsl_mamba_drop_path,
            fsl_mamba_perturb_sigma=fsl_mamba_perturb_sigma,
        )
        self.encoder = self.backbone_model.encode
        self.entropy_reg_weight = float(entropy_reg_weight)
        self.head = RADAFewShotHead(
            feat_dim=hidden_dim,
            tau_r=tau_r,
            lambda_proto=lambda_proto,
            gamma_disp=gamma_disp,
            eps=eps,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_head=reliability_head,
            l2_normalize=l2_normalize,
            use_reliability=use_reliability,
            use_dispersion_metric=use_dispersion_metric,
            query_conditioned=query_conditioned,
            use_residual_anchor=use_residual_anchor,
            use_shrinkage=use_shrinkage,
            disp_clamp_max=disp_clamp_max,
            use_evidence_bound=use_evidence_bound,
            evidence_temperature=evidence_temperature,
            min_reliability_mix=min_reliability_mix,
            dispersion_inflation=dispersion_inflation,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return pooled_episode_features(self.encoder(x))

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone_model.get_features(images)

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_query, channels, height, width, way_num = self.backbone_model.validate_episode_inputs(
            query,
            support,
        )
        del channels, height, width
        shot_num = support.shape[2]

        query_flat = query.reshape(batch_size * num_query, *query.shape[-3:])
        support_flat = support.reshape(batch_size * way_num * shot_num, *support.shape[-3:])

        query_feat = self.extract_features(query_flat).reshape(batch_size, num_query, -1)
        support_feat = self.extract_features(support_flat).reshape(batch_size, way_num, shot_num, -1)
        return query_feat, support_feat

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        query_feat, support_feat = self._encode_episode(query, support)
        logits, aux = self.head(support_feat=support_feat, query_feat=query_feat)

        batch_size, num_query, way_num = logits.shape
        flat_logits = logits.reshape(batch_size * num_query, way_num)

        if not return_aux and self.entropy_reg_weight <= 0.0:
            return flat_logits

        outputs = {
            "logits": flat_logits,
            "episode_logits": logits,
            "alpha": aux["alpha"],
            "proto": aux["proto"],
            "disp": aux["disp"],
            "mu": aux["mu"],
            "raw_scatter": aux["raw_scatter"],
            "support_dispersion": aux["raw_scatter"],
            "reliability_logits": aux["reliability_logits"],
            "reliability_mix_tensor": aux["reliability_mix"],
            "delta": aux["delta"],
            "global_disp": aux["global_disp"],
            "alpha_entropy_tensor": aux["alpha_entropy"],
            "alpha_max_tensor": aux["alpha_max"],
            "effective_support_size_tensor": aux["effective_support_size"],
            "dispersion_inflation_tensor": aux["dispersion_inflation"],
            "prototype_shift_norm_tensor": aux["prototype_shift_norm"],
            "alpha_entropy": aux["alpha_entropy"].mean().detach(),
            "alpha_max_mean": aux["alpha_max"].mean().detach(),
            "reliability_mix": aux["reliability_mix"].mean().detach(),
            "effective_support_size": aux["effective_support_size"].mean().detach(),
            "dispersion_inflation": aux["dispersion_inflation"].mean().detach(),
            "dispersion_mean": aux["disp"].mean().detach(),
            "dispersion_min": aux["disp"].min().detach(),
            "prototype_shift_norm": aux["prototype_shift_norm"].mean().detach(),
        }
        if self.entropy_reg_weight > 0.0:
            entropy_loss = -self.entropy_reg_weight * aux["alpha_entropy"].mean()
            outputs["entropy_loss"] = entropy_loss.detach()
            outputs["aux_loss"] = entropy_loss if self.training else flat_logits.new_zeros(())
        return outputs
