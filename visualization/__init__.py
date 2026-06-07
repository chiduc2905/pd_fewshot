"""Feature visualization module for few-shot learning models."""

from .feature_visualizer import (
    FeatureVisualizer,
    extract_features,
    visualize_feature_maps,
    create_activation_grid,
)
from .noise_diagnostics import (
    infer_token_hw,
    extract_focus_maps,
    compute_focus_metrics,
    compute_scalogram_statistics,
    compute_support_episode_distribution,
    export_dataset_noise_profile,
    export_episode_q1_figure,
    export_support_distribution_figure,
    export_uot_evidence_figure,
    score_uot_evidence_candidate,
)

__all__ = [
    'FeatureVisualizer',
    'extract_features',
    'visualize_feature_maps',
    'create_activation_grid',
    'infer_token_hw',
    'extract_focus_maps',
    'compute_focus_metrics',
    'compute_scalogram_statistics',
    'compute_support_episode_distribution',
    'export_dataset_noise_profile',
    'export_episode_q1_figure',
    'export_support_distribution_figure',
    'export_uot_evidence_figure',
    'score_uot_evidence_candidate',
]
