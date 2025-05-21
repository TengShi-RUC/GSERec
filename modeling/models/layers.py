import logging

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNEncoder(nn.Module):

    def __init__(self, N_count, M_count, adj_mat, num_layers):
        super(LightGCNEncoder, self).__init__()
        self.N_count = N_count
        self.M_count = M_count
        self.num_layers = num_layers

        self.norm_adj_mat = self.normalized_adj_single(adj_mat)

        logging.info('use sparse adj')
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(
            self.norm_adj_mat).cuda()

    def forward(self, all_N_emb, all_M_emb=None):
        if all_M_emb is None:
            ego_embs = all_N_emb
        else:
            ego_embs = torch.cat([all_N_emb, all_M_emb], dim=0)
        all_embs = [ego_embs]

        for k in range(self.num_layers):
            ego_embs = torch.spmm(self.sparse_norm_adj, ego_embs)
            all_embs += [ego_embs]

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1)

        N_all_embs = all_embs[:self.N_count, :]

        if all_M_emb is None:
            M_all_embs = None
        else:
            M_all_embs = all_embs[self.N_count:, :]

        return N_all_embs, M_all_embs

    @staticmethod
    def normalized_adj_single(adj: sp.dok_matrix):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1)) + 1e-10

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

        return bi_lap.tocoo().tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col], dtype=np.int64))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    @staticmethod
    def _convert_sp_tensor_to_sp_mat(X: torch.Tensor):
        X = X.cpu()
        data = X.coalesce().values()
        indices = X.coalesce().indices()
        shape = X.shape
        return sp.coo_matrix((data.numpy(), indices.numpy()), shape=shape)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 emb_size,
                 num_heads,
                 num_layers,
                 dropout,
                 add_pos=False,
                 his_len=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            transformerEncoderLayer, num_layers=num_layers)

        self.add_pos = add_pos
        if add_pos:
            assert his_len is not None
            self.pos_embedding = PositionalEmbedding(his_len, emb_size)

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                mask: torch.Tensor = None,
                used_mask: torch.Tensor = None):
        src_key_padding_mask = src_key_padding_mask.bool()
        mask = mask.bool() if mask is not None else mask
        used_mask = used_mask.bool() if used_mask is not None else used_mask

        if self.add_pos:
            src = src + self.pos_embedding(src)

        result_encoded = torch.zeros_like(src).to(src.device)

        if used_mask is None:
            none_zero_his = (~src_key_padding_mask).sum(dim=1) > 0
        else:
            none_zero_his = used_mask

        if none_zero_his.sum() <= 0:
            return result_encoded

        sub_src = src[none_zero_his]
        sub_src_key_padding_mask = src_key_padding_mask[none_zero_his]

        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).expand(src.shape[0], -1, -1)

            sub_mask = mask[none_zero_his]
            sub_mask_expand = sub_mask.unsqueeze(1).expand(
                (-1, self.num_heads, -1, -1)).reshape(
                    (-1, src.size(1), src.size(1)))
            sub_his_encoded = self.transformer_encoder(
                src=sub_src,
                src_key_padding_mask=sub_src_key_padding_mask,
                mask=sub_mask_expand)
        else:
            sub_his_encoded = self.transformer_encoder(
                src=sub_src, src_key_padding_mask=sub_src_key_padding_mask)

        result_encoded[none_zero_his] = sub_his_encoded

        return result_encoded


class TransformerDecoder(nn.Module):

    def __init__(self, emb_size, num_heads, num_layers, dropout):
        super().__init__()
        transformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size,
            dropout=dropout,
            batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder(
            transformerDecoderLayer, num_layers=num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        tgt_key_padding_mask = tgt_key_padding_mask.bool()
        memory_key_padding_mask = memory_key_padding_mask.bool()

        result_encoded = torch.zeros_like(tgt).to(tgt.device)

        none_zero_his = ((~tgt_key_padding_mask).sum(dim=1) > 0) & (
            (~memory_key_padding_mask).sum(dim=1) > 0)

        if none_zero_his.sum() <= 0:
            return result_encoded

        sub_tgt = tgt[none_zero_his]
        sub_tgt_key_padding_mask = tgt_key_padding_mask[none_zero_his]

        sub_memory = memory[none_zero_his]
        sub_memory_key_padding_mask = memory_key_padding_mask[none_zero_his]

        sub_decoded = self.transformer_decoder(
            tgt=sub_tgt,
            memory=sub_memory,
            tgt_key_padding_mask=sub_tgt_key_padding_mask,
            memory_key_padding_mask=sub_memory_key_padding_mask)

        result_encoded[none_zero_his] = sub_decoded

        return result_encoded


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class FullyConnectedLayer(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_unit,
                 batch_norm=False,
                 activation='relu',
                 sigmoid=False,
                 dropout=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i + 1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)
