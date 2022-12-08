import torch
from torch import nn

class FFN_block(nn.Sequential):
    """
    Feed forward block
    """
    def __init__(self, emb_in: int, emb_out: int):
        super().__init__(
            nn.ReLU(),
            nn.Linear(emb_in, emb_out),
        )

class head (nn.Sequential):
    """
    A ReLU stack that takes in a tensor of (b x in_size) and outputs (b x outsize)
    """
    def __init__(self, in_size: int = 128, emb_size: int = 256, out_size: int = 6, depth: int = 3):
        super().__init__(*([nn.Linear(in_size, emb_size),]+[FFN_block(emb_size, emb_size) for _ in range(depth)]+[nn.Linear(emb_size, out_size)]))