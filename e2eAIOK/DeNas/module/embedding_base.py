import torch.nn as nn


class EmbeddingBase(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, blank_id=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=blank_id)

    def forward(self, x):
        raise NotImplementedError
