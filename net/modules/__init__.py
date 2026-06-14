"""Helper modules for pulse_fewshot consensus-slot models."""

from .crs_marginal import CrossReferencedSelectiveMarginal, SimpleBidirectionalScan
from .token_attention_marginal import TokenAttentionMarginal
from .cata import CATA
from .hierarchical_consensus_slot_mamba import AttentiveTokenPool, IntraImageConsensusMambaEncoder
from .hierarchical_reliability_consensus import (
    DualConsensusReliabilityHead,
    ReliabilityCoupledTokenSWHead,
    ShotConditionedTokenAdapter,
)
from .hierarchical_set_v2 import ConsensusClassAggregator, SetConditionedShotRefiner
from .mspta import MSPTATokenizer
from .mutual_evidence_attention import MutualEvidenceAttentionMarginal
from .spatial_context_injection import SpatialContextInjection
from .verified_region_matching import VerifiedRegionMatchingUOT

__all__ = [
    "AttentiveTokenPool",
    "CATA",
    "CrossReferencedSelectiveMarginal",
    "IntraImageConsensusMambaEncoder",
    "SimpleBidirectionalScan",
    "DualConsensusReliabilityHead",
    "ReliabilityCoupledTokenSWHead",
    "ShotConditionedTokenAdapter",
    "ConsensusClassAggregator",
    "MSPTATokenizer",
    "MutualEvidenceAttentionMarginal",
    "SetConditionedShotRefiner",
    "SpatialContextInjection",
    "VerifiedRegionMatchingUOT",
]

