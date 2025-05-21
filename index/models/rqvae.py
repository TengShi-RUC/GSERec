import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):

    def __init__(
        self,
        src_dim,
        rec_dim,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        src_n_e_list=None,
        rec_n_e_list=None,
        e_dim=None,
        sk_epsilons=None,
        cl_temp=1,
        cl_weight=0,
        batch_size=None,
    ):
        super(RQVAE, self).__init__()
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.src_dim = src_dim
        self.rec_dim = rec_dim

        self.src_n_e_list = src_n_e_list
        self.rec_n_e_list = rec_n_e_list

        self.e_dim = e_dim
        self.hidden_dim = e_dim
        self.sk_epsilons = sk_epsilons

        self.src_encode_layer_dims = [self.src_dim
                                      ] + self.layers + [self.hidden_dim]
        self.src_encoder = MLPLayers(layers=self.src_encode_layer_dims,
                                     dropout=self.dropout_prob,
                                     bn=self.bn)

        self.src_decoder_layer_dims = self.src_encode_layer_dims[::-1]
        self.src_decoder = MLPLayers(layers=self.src_decoder_layer_dims,
                                     dropout=self.dropout_prob,
                                     bn=self.bn)

        self.rec_encode_layer_dims = [self.rec_dim
                                      ] + self.layers + [self.hidden_dim]
        self.rec_encoder = MLPLayers(layers=self.rec_encode_layer_dims,
                                     dropout=self.dropout_prob,
                                     bn=self.bn)

        self.rec_decoder_layer_dims = self.rec_encode_layer_dims[::-1]
        self.rec_decoder = MLPLayers(layers=self.rec_decoder_layer_dims,
                                     dropout=self.dropout_prob,
                                     bn=self.bn)

        self.cl_temp = cl_temp
        self.cl_weight = cl_weight
        self.infoNCE = InfoNCE(batch_size=batch_size,
                               infoNCE_temp=self.cl_temp)

        self.rq = ResidualVectorQuantizer(
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
            src_n_e_list=self.src_n_e_list,
            rec_n_e_list=self.rec_n_e_list,
            e_dim=self.e_dim,
            sk_epsilons=self.sk_epsilons,
        )

    def forward(self, x, use_sk=True):
        src_x = self.src_encoder(x['src'])
        rec_x = self.rec_encoder(x['rec'])
        x = torch.cat([src_x, rec_x], dim=-1)

        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)

        src_x_q, rec_x_q = x_q
        src_x_q = self.src_decoder(src_x_q)
        rec_x_q = self.rec_decoder(rec_x_q)

        out = {"src": src_x_q, "rec": rec_x_q}

        if self.cl_weight > 0:
            assert self.src_dim and self.rec_dim

            cl_loss = self.infoNCE(src_x, rec_x)

        else:
            cl_loss = 0

        return out, rq_loss, indices, cl_loss

    @torch.no_grad()
    def get_indices(self, x, use_sk=False):
        src_x = self.src_encoder(x['src'])
        rec_x = self.rec_encoder(x['rec'])
        x = torch.cat([src_x, rec_x], dim=-1)

        _, _, indices = self.rq(x, use_sk=use_sk)

        return indices

    def compute_loss(self, out, quant_loss, cl_loss, xs=None):

        if self.loss_type == 'mse':
            loss_fn = F.mse_loss
        elif self.loss_type == 'l1':
            loss_fn = F.l1_loss
        else:
            raise ValueError('incompatible loss type')

        src_loss = loss_fn(out['src'], xs['src'], reduction='mean')

        rec_loss = loss_fn(out['rec'], xs['rec'], reduction='mean')

        loss_recon = src_loss + rec_loss

        loss_total = loss_recon + self.quant_loss_weight * quant_loss + self.cl_weight * cl_loss

        return loss_total, loss_recon, quant_loss, cl_loss

    def count_variables(self) -> int:
        total_parameters = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                num_p = p.numel()
                total_parameters += num_p

        return total_parameters


class InfoNCE(nn.Module):

    def __init__(self, batch_size, infoNCE_temp):
        super().__init__()
        self.batch_size = batch_size
        self.infoNCE_temp = infoNCE_temp

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = nn.Parameter(self.mask_correlated_samples(
            self.batch_size),
                                         requires_grad=False)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, emb_1: torch.Tensor, emb_2: torch.Tensor):
        batch_size = emb_1.size(0)
        N = 2 * batch_size

        emb_1 = F.normalize(emb_1, p=2, dim=-1)
        emb_2 = F.normalize(emb_2, p=2, dim=-1)

        z = torch.cat([emb_1, emb_2], dim=0)
        sim = torch.mm(z, z.T)
        sim = sim / self.infoNCE_temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss
