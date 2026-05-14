"""Standalone Ours model and contribution ablations.

The full design is intentionally treated as one coherent model:
J-ECOT-M2/SB-ECOT with UOT fixed to the paper-facing defaults.  Ours
``full`` enables Episode-Gated Shrinkage Marginals (EGSM) and turns off
MEA/CCDM/CRS for the ECOT token priors.
Contribution ablations swap only high-level design choices while leaving the
full model path untouched.
"""

from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn.functional as F

from net.hrot_fsl import HROTFSLResult
from net.jecot_m2 import JECOTM2


OURS_ABLATIONS = frozenset({"full", "full_ot", "no_egsm", "gap"})


def normalize_ours_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "full",
        "ours": "full",
        "uot": "full",
        "balanced_ot": "full_ot",
        "balanced": "full_ot",
        "ot": "full_ot",
        "uniform_evidence": "no_egsm",
        "uniform": "no_egsm",
        "no_adaptive_evidence": "no_egsm",
        "no_evidence": "no_egsm",
        "proto": "gap",
        "prototype": "gap",
        "global_prototype": "gap",
        "gap_proto": "gap",
    }
    name = aliases.get(name, name)
    if name not in OURS_ABLATIONS:
        raise ValueError(f"Unsupported Ours ablation: {value}. Expected one of {sorted(OURS_ABLATIONS)}")
    return name


def apply_ours_design_defaults(kwargs: dict, ablation: str) -> dict:
    """Apply the standalone Ours design contract to HROT/M2 constructor kwargs."""
    kwargs = dict(kwargs)

    # Full Ours: local descriptors + one UOT budget, with the M2 mass score removed.
    kwargs.setdefault("use_cata", False)
    kwargs.setdefault("cata_num_anchors", 8)
    kwargs.setdefault("cata_num_heads", 4)
    kwargs.setdefault("cata_attn_dropout", 0.0)
    kwargs["ecot_rho_bank"] = "0.8"
    kwargs["ecot_base_rho"] = 0.8
    kwargs["ecot_transport_mode"] = "unbalanced"
    # Default Ours score is -cost only. If the caller already enabled cost/mass
    # (CLI / run_all_experiments --m2_cost_per_mass), do not overwrite it here —
    # otherwise cost/mass appears to have no effect vs baseline.
    if kwargs.get("ecot_m2_cost_per_mass_score"):
        kwargs["ecot_m2_ablate_threshold_mass"] = False
        kwargs.setdefault("ecot_m2_cost_per_mass_detach_mass", True)
    elif kwargs.get("ecot_m2_ablate_threshold_mass") is False:
        kwargs["ecot_m2_cost_per_mass_score"] = False
    else:
        kwargs["ecot_m2_ablate_threshold_mass"] = True
        kwargs["ecot_m2_cost_per_mass_score"] = False
        kwargs["ecot_m2_cost_per_mass_detach_mass"] = False
    kwargs["ecot_m2_use_aqm"] = False
    kwargs["ecot_m2_tau_aqm"] = 1.0
    kwargs["ecot_m2_use_swts"] = False
    kwargs["ecot_m2_swts_temp"] = 1.0
    kwargs["ecot_enable_ccdm_marginal"] = False
    kwargs["ecot_enable_mea_marginal"] = False
    kwargs["ecot_enable_crs_marginal"] = False
    if ablation in {"full", "full_ot"}:
        kwargs.setdefault("ecot_enable_egsm", True)

    if ablation == "full_ot":
        kwargs["ecot_rho_bank"] = "1.0"
        kwargs["ecot_base_rho"] = 1.0
        kwargs["ecot_transport_mode"] = "balanced"
    elif ablation == "no_egsm":
        # Full Ours without episode-gated shrinkage marginals.
        kwargs["ecot_m2_use_aqm"] = False
        kwargs["ecot_m2_use_swts"] = False
        kwargs["ecot_enable_egsm"] = False
        kwargs["ecot_enable_ccdm_marginal"] = False
        kwargs["ecot_enable_mea_marginal"] = False
        kwargs["ecot_enable_crs_marginal"] = False
    elif ablation == "gap":
        # GAP replaces the local descriptor grid with one global token per
        # image; EGSM is vacuous on a one-token measure, so keep this control
        # focused on local-vs-global evidence.
        kwargs["ecot_enable_egsm"] = False

    return kwargs


def _bool_config(value: bool | str) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")
    return bool(value)


def _resolve_dm_debug(value: bool | str, use_differential_mode: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return bool(use_differential_mode)
    return _bool_config(value) and bool(use_differential_mode)


def _infer_token_hw(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    root = int(math.sqrt(num_tokens))
    best = (1, num_tokens)
    best_gap = num_tokens - 1
    for height in range(1, root + 1):
        if num_tokens % height != 0:
            continue
        width = num_tokens // height
        gap = abs(width - height)
        if gap < best_gap:
            best = (height, width)
            best_gap = gap
    return best


class OursM2(JECOTM2):
    """Paper-facing Ours entrypoint plus coarse contribution ablations."""

    def __init__(
        self,
        *args,
        ours_ablation: str = "full",
        use_differential_mode: bool = False,
        dm_alpha: float = 0.0,
        dm_debug: bool | str = "auto",
        dm_debug_dir: str = "results/dmt_debug",
        dm_debug_max_episodes: int = 5,
        **kwargs,
    ) -> None:
        if not 0.0 <= float(dm_alpha) <= 1.0:
            raise ValueError("dm_alpha must be in [0, 1]")
        if int(dm_debug_max_episodes) < 0:
            raise ValueError("dm_debug_max_episodes must be non-negative")
        self.ours_ablation = normalize_ours_ablation(ours_ablation)
        self.use_differential_mode = _bool_config(use_differential_mode)
        self.dm_alpha = float(dm_alpha)
        self.dm_debug = _resolve_dm_debug(dm_debug, self.use_differential_mode)
        self.dm_debug_dir = str(dm_debug_dir)
        self.dm_debug_max_episodes = int(dm_debug_max_episodes)
        self._dm_debug_count = 0
        self._last_dm_diagnostics: dict[str, torch.Tensor] | None = None
        kwargs = apply_ours_design_defaults(kwargs, self.ours_ablation)
        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )
        self.register_buffer("dm_global_template", torch.empty(0), persistent=False)
        self.register_buffer("dm_template_count", torch.tensor(0.0), persistent=False)

    @property
    def uses_ours_gap_control(self) -> bool:
        return self.ours_ablation == "gap"

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if not self.uses_ours_gap_control:
            return super()._encode_images(images)

        feature_map = self.encode(images)
        gap = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1).unsqueeze(1)
        projected = self.token_projector(gap)
        euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
        return euclidean_tokens, hyperbolic_tokens, (1, 1)

    def _update_differential_mode_template(self, episode_template: torch.Tensor) -> None:
        if not self.training:
            return
        with torch.no_grad():
            template = episode_template.detach()
            if tuple(self.dm_global_template.shape) != tuple(template.shape):
                self.dm_global_template = template.clone()
                self.dm_template_count = template.new_tensor(1.0)
                return
            count = self.dm_template_count.to(device=template.device, dtype=template.dtype)
            next_count = count + 1.0
            updated = self.dm_global_template.to(device=template.device, dtype=template.dtype)
            updated = updated + (template - updated) / next_count
            self.dm_global_template = updated.detach()
            self.dm_template_count = next_count.detach()

    def _differential_mode_template(self, support_tokens: torch.Tensor) -> torch.Tensor:
        episode_template = support_tokens.mean(dim=0)
        self._update_differential_mode_template(episode_template)
        if (
            self.training
            or self.dm_alpha <= 0.0
            or self.dm_global_template.numel() == 0
            or tuple(self.dm_global_template.shape) != tuple(episode_template.shape)
        ):
            return episode_template
        global_template = self.dm_global_template.to(
            device=episode_template.device,
            dtype=episode_template.dtype,
        )
        alpha = float(self.dm_alpha)
        return alpha * global_template + (1.0 - alpha) * episode_template

    def _apply_differential_mode_to_cost_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_differential_mode:
            return query_tokens, support_tokens
        if query_tokens.dim() != 3 or support_tokens.dim() != 3:
            raise ValueError(
                "Ours differential mode expects query/support cost tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumSupport, Tokens, Dim)"
            )
        if query_tokens.shape[-2:] != support_tokens.shape[-2:]:
            raise ValueError(
                "Ours differential mode requires aligned spatial token grids, "
                f"got {tuple(query_tokens.shape[-2:])} vs {tuple(support_tokens.shape[-2:])}"
            )

        template = self._differential_mode_template(support_tokens)
        query_diff = query_tokens - template.unsqueeze(0)
        support_diff = support_tokens - template.unsqueeze(0)
        torch._assert(query_diff.shape == query_tokens.shape, "DMT changed query token shape")
        torch._assert(support_diff.shape == support_tokens.shape, "DMT changed support token shape")
        self._last_dm_diagnostics = self._build_differential_mode_diagnostics(
            template,
            query_tokens,
            support_tokens,
            query_diff,
            support_diff,
        )
        self._maybe_export_differential_mode_debug(template, self._last_dm_diagnostics)
        return query_diff, support_diff

    def _build_differential_mode_diagnostics(
        self,
        template: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_diff: torch.Tensor,
        support_diff: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = float(self.eps)
        query_residual_ratio = query_diff.norm() / query_tokens.norm().clamp_min(eps)
        support_residual_ratio = support_diff.norm() / support_tokens.norm().clamp_min(eps)
        residual_ratio = 0.5 * (query_residual_ratio + support_residual_ratio)
        token_norm = template.norm(dim=-1)
        return {
            "dm_mu_norm": token_norm.mean(),
            "dm_mu_max_norm": token_norm.max(),
            "dm_mu_support_ratio": template.norm() / support_tokens.norm().clamp_min(eps),
            "dm_query_residual_ratio": query_residual_ratio,
            "dm_support_residual_ratio": support_residual_ratio,
            "dm_residual_ratio": residual_ratio,
            "dm_alpha": template.new_tensor(float(self.dm_alpha)),
            "dm_template_count": self.dm_template_count.to(device=template.device, dtype=template.dtype),
            "dm_mu_token_norm": token_norm.detach(),
        }

    def _maybe_export_differential_mode_debug(
        self,
        template: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        if not self.dm_debug:
            return
        if self.dm_debug_max_episodes > 0 and self._dm_debug_count >= self.dm_debug_max_episodes:
            return
        self._dm_debug_count += 1
        debug_idx = self._dm_debug_count
        try:
            os.makedirs(self.dm_debug_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug disabled: cannot create {self.dm_debug_dir}: {exc}")
            return

        token_norm = diagnostics["dm_mu_token_norm"].detach().float().cpu()
        token_hw = _infer_token_hw(int(token_norm.numel()))
        heatmap_path = os.path.join(self.dm_debug_dir, f"dmt_mu_heatmap_{debug_idx:04d}.png")

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            heatmap = token_norm.reshape(*token_hw).numpy()
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            image = ax.imshow(heatmap, cmap="magma")
            ax.set_title(
                "Ours DMT common component | "
                f"mu_norm={float(diagnostics['dm_mu_norm'].detach().cpu()):.4f}"
            )
            ax.set_xlabel("token x")
            ax.set_ylabel("token y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - defensive debug path
            heatmap_path = f"failed: {exc}"

        csv_path = os.path.join(self.dm_debug_dir, "dmt_debug_metrics.csv")
        write_header = not os.path.exists(csv_path)
        row = {
            "episode": debug_idx,
            "training": int(self.training),
            "dm_alpha": float(diagnostics["dm_alpha"].detach().cpu()),
            "dm_template_count": float(diagnostics["dm_template_count"].detach().cpu()),
            "dm_mu_norm": float(diagnostics["dm_mu_norm"].detach().cpu()),
            "dm_mu_max_norm": float(diagnostics["dm_mu_max_norm"].detach().cpu()),
            "dm_mu_support_ratio": float(diagnostics["dm_mu_support_ratio"].detach().cpu()),
            "dm_query_residual_ratio": float(diagnostics["dm_query_residual_ratio"].detach().cpu()),
            "dm_support_residual_ratio": float(diagnostics["dm_support_residual_ratio"].detach().cpu()),
            "dm_residual_ratio": float(diagnostics["dm_residual_ratio"].detach().cpu()),
            "heatmap_path": heatmap_path,
        }
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug metrics skipped: cannot write {csv_path}: {exc}")
            return

        print(
            "[Ours-DMT] "
            f"episode={debug_idx} "
            f"mu_norm={row['dm_mu_norm']:.6f} "
            f"mu_support_ratio={row['dm_mu_support_ratio']:.6f} "
            f"residual_ratio={row['dm_residual_ratio']:.6f} "
            f"q_residual={row['dm_query_residual_ratio']:.6f} "
            f"s_residual={row['dm_support_residual_ratio']:.6f} "
            f"heatmap={heatmap_path}"
        )

    def _ground_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens, class_tokens = self._apply_differential_mode_to_cost_tokens(
            query_tokens,
            class_tokens,
        )
        return super()._ground_cost(query_tokens, class_tokens)

    def _forward_episode(self, *args, **kwargs) -> torch.Tensor | dict[str, torch.Tensor]:
        self._last_dm_diagnostics = None
        outputs = super()._forward_episode(*args, **kwargs)
        if isinstance(outputs, dict) and self._last_dm_diagnostics is not None:
            outputs.update(self._last_dm_diagnostics)
        return outputs

    def _stack_outputs(self, batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = super()._stack_outputs(batch_outputs)
        for key in (
            "dm_mu_norm",
            "dm_mu_max_norm",
            "dm_mu_support_ratio",
            "dm_query_residual_ratio",
            "dm_support_residual_ratio",
            "dm_residual_ratio",
            "dm_alpha",
            "dm_template_count",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "dm_mu_token_norm" in batch_outputs[0]:
            stacked["dm_mu_token_norm"] = torch.stack(
                [item["dm_mu_token_norm"] for item in batch_outputs],
                dim=0,
            )
        return stacked

__all__ = ["OursM2", "apply_ours_design_defaults", "OURS_ABLATIONS", "normalize_ours_ablation"]
