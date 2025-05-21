import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(
        self,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        src_n_e_list=None,
        rec_n_e_list=None,
        e_dim=None,
        sk_epsilons=None,
    ):
        super().__init__()
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.src_n_e_list = src_n_e_list
        self.rec_n_e_list = rec_n_e_list

        self.e_dim = e_dim
        self.sk_epsilons = sk_epsilons

        self.num_quantizers = len(src_n_e_list)
        assert len(src_n_e_list) == len(rec_n_e_list)

        self.vq_layers_src = nn.ModuleList([
            VectorQuantizer(n_e,
                            e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(src_n_e_list, sk_epsilons)
        ])
        self.vq_layers_rec = nn.ModuleList([
            VectorQuantizer(n_e,
                            e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(rec_n_e_list, sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        residual = x

        src_losses = []
        src_indices = []
        residual_src = residual[:, :self.e_dim]
        src_x_q = 0
        for quantizer in self.vq_layers_src:
            x_res, loss, indices = quantizer(residual_src, use_sk=use_sk)
            residual_src = residual_src - x_res
            src_x_q = src_x_q + x_res

            src_losses.append(loss)
            src_indices.append(indices)

        rec_losses = []
        rec_indices = []
        residual_rec = residual[:, self.e_dim:]
        rec_x_q = 0
        for quantizer in self.vq_layers_rec:
            x_res, loss, indices = quantizer(residual_rec, use_sk=use_sk)
            residual_rec = residual_rec - x_res
            rec_x_q = rec_x_q + x_res

            rec_losses.append(loss)
            rec_indices.append(indices)

        all_losses = src_losses + rec_losses
        mean_losses = torch.stack(all_losses).mean()

        src_indices = torch.stack(src_indices, dim=-1)
        rec_indices = torch.stack(rec_indices, dim=-1)

        return (src_x_q, rec_x_q), mean_losses, (src_indices, rec_indices)
