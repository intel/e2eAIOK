import torch.nn.functional as F

from e2eAIOK.DeNas.module.embedding_base import EmbeddingBase


class Embedding(EmbeddingBase):
    """
    Computes an embedding x = wx.
    """

    def __init__(self, num_embeddings, embedding_dim, blank_id=0):

        super().__init__(num_embeddings, embedding_dim, blank_id=0)

    def forward(self, x):
        return F.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
