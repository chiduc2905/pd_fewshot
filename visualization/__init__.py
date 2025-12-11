"""Feature visualization module for few-shot learning models."""

from .feature_visualizer import (
    FeatureVisualizer,
    extract_features,
    visualize_feature_maps,
    create_activation_grid,
)

__all__ = [
    'FeatureVisualizer',
    'extract_features',
    'visualize_feature_maps',
    'create_activation_grid',
]
