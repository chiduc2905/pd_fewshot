"""Backward-compatible shim for the renamed few-shot Mamba encoder."""

from .fsl_mamba_encoder import FSLMambaEncoder, FSLMambaEncoder as SlimMambaEncoder

__all__ = ["FSLMambaEncoder", "SlimMambaEncoder"]
