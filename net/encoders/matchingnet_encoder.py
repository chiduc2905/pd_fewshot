"""Matching Networks encoder."""

from net.encoders.protonet_encoder import Conv64F_Paper_Encoder


class MatchingNetEncoder(Conv64F_Paper_Encoder):
    """Conv4 embedding used by Matching Networks benchmarks."""

    def __init__(self, image_size=64):
        super().__init__(image_size=image_size)




