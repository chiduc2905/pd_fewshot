"""Primary UBT-FSL import path.

The implementation lives in `net.cbcr_fsl` to preserve legacy checkpoints and
scripts. New code should import `UBTFSL` from this module.
"""

from net.cbcr_fsl import (
    CBCRFSL,
    CBCRFSLResult,
    TokenMeasure,
    UBT_ABLATION_MODES,
    UBTFSL,
    UBTFSLResult,
    UncertainBarycentricClass,
)

__all__ = [
    "TokenMeasure",
    "UncertainBarycentricClass",
    "UBT_ABLATION_MODES",
    "UBTFSL",
    "UBTFSLResult",
    "CBCRFSL",
    "CBCRFSLResult",
]
