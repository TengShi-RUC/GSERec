import logging
from typing import List

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from modules import const

from .BaseModel import BaseModel
from .Inputs import *
from .layers import (FullyConnectedLayer, LightGCNEncoder, TransformerDecoder,
                     TransformerEncoder)


class GSERec(BaseModel):

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=2)

        parser.add_argument('--num_gnn_layers', type=int, default=2)

        parser.add_argument('--user_rec_index',
                            type=str,
                            default='')
        parser.add_argument('--user_src_index',
                            type=str,
                            default='')

        parser.add_argument('--user_cl_temp', type=float, default=0.1)
        parser.add_argument('--user_cl_weight', type=float, default=0.1)

        parser.add_argument('--code_his_cl_temp', type=float, default=0.1)
        parser.add_argument('--code_his_cl_weight', type=float, default=0.01)

        parser.add_argument('--pred_hid_units',
                            type=List,
                            default=[200, 80, 1])

        return BaseModel.parse_model_args(parser)

    def __init__(self, args, all_vocabs):
        super().__init__(args, all_vocabs)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_gnn_layers = args.num_gnn_layers

        self.rec_his_transformer_layer = TransformerEncoder(
            emb_size=self.item_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            add_pos=True,
            his_len=const.max_rec_his_len)

        self.src_his_transformer_layer = TransformerEncoder(
            emb_size=self.item_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            add_pos=True,
            his_len=const.max_src_session_his_len)

        self.rec_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)
        self.src_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)

        self.build_u_code_graph()

        self.user_cl_weight = args.user_cl_weight
        self.user_cl_temp = args.user_cl_temp

        self.code_his_cl_weight = args.code_his_cl_weight
        self.code_his_cl_temp = args.code_his_cl_temp

        self.user_cl = EmbCL(batch_size=self.batch_size,
                             hidden_dim=const.final_emb_size,
                             device=self.device,
                             infoNCE_temp=self.user_cl_temp)

        self.rec_code_fusion = TransformerDecoder(emb_size=self.item_size,
                                                  num_heads=self.num_heads,
                                                  num_layers=self.num_layers,
                                                  dropout=self.dropout)
        self.src_code_fusion = TransformerDecoder(emb_size=self.item_size,
                                                  num_heads=self.num_heads,
                                                  num_layers=self.num_layers,
                                                  dropout=self.dropout)

        self.rec_code_his_cl = HisCL(batch_size=self.batch_size,
                                     hidden_dim=const.final_emb_size,
                                     device=self.device,
                                     infoNCE_temp=self.code_his_cl_temp)
        self.src_code_his_cl = HisCL(batch_size=self.batch_size,
                                     hidden_dim=const.final_emb_size,
                                     device=self.device,
                                     infoNCE_temp=self.code_his_cl_temp)

        self.hidden_unit = args.pred_hid_units

        input_dim = 3 * self.item_size + 2 * self.user_size

        self.rec_fc_layer = FullyConnectedLayer(input_size=input_dim,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=False,
                                                activation='relu',
                                                dropout=self.dropout)

        self._init_weights()
        self.to(self.device)

    def init_emb(self, args, all_vocabs):
        query_emb = QueryFeat(self.device,
                              query_vocab=all_vocabs['query_vocab'],
                              dropout=self.dropout)

        self.user_emb = UserFeat(all_vocabs['user_vocab'],
                                 self.device,
                                 dropout=self.dropout)

        self.user_code_emb = UserFeatCode(device=self.device,
                                          rec_index_path=args.user_rec_index,
                                          src_index_path=args.user_src_index)

        item_emb = ItemFeat(all_vocabs['item_vocab'],
                            self.device,
                            query_emb,
                            dropout=self.dropout)

        self.session_emb = SrcSessionFeat(
            query_feat=query_emb,
            item_feat=item_emb,
            session_vocab=all_vocabs['session_vocab'],
            device=self.device)

    def build_u_code_adjmat(self, user_code_pair, code_num):
        adj_mat = sp.dok_matrix(
            (const.user_id_num + code_num, const.user_id_num + code_num),
            dtype=np.float32)
        for user, code in user_code_pair:
            adj_mat[int(user), int(code) + const.user_id_num] = 1
            adj_mat[int(code) + const.user_id_num, int(user)] = 1

        return adj_mat

    def build_u_code_graph(self):
        logging.info('Building u-code graph')
        code_num = len(self.user_code_emb.code2id)

        rec_adj_mat = self.build_u_code_adjmat(
            self.user_code_emb.user_rec_code_pair, code_num)
        self.u_c_rec_encoder = LightGCNEncoder(N_count=const.user_id_num,
                                               M_count=code_num,
                                               adj_mat=rec_adj_mat,
                                               num_layers=self.num_gnn_layers)

        src_adj_mat = self.build_u_code_adjmat(
            self.user_code_emb.user_src_code_pair, code_num)
        self.u_c_src_encoder = LightGCNEncoder(N_count=const.user_id_num,
                                               M_count=code_num,
                                               adj_mat=src_adj_mat,
                                               num_layers=self.num_gnn_layers)

    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        mean_click_item_emb = torch.sum(torch.mul(
            q_click_item_emb, click_item_mask.unsqueeze(-1)),
                                        dim=-2)  # batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (torch.max(
            click_item_mask.sum(-1, keepdim=True),
            torch.ones_like(click_item_mask.sum(-1, keepdim=True))))
        query_his_emb = query_emb
        click_item_his_emb = mean_click_item_emb

        return query_his_emb + click_item_his_emb

    def get_user_emb(self, user):
        all_user_embs = self.user_emb.get_all_user_emb()
        all_user_code_embs = self.user_code_emb.get_all_user_code_emb()

        user_rec_embs_all, rec_code_emb_all = self.u_c_rec_encoder(
            all_user_embs, all_user_code_embs)
        user_rec_code = self.user_code_emb.user2rec_code[user]

        user_src_embs_all, src_code_emb_all = self.u_c_src_encoder(
            all_user_embs, all_user_code_embs)
        user_src_code = self.user_code_emb.user2src_code[user]

        user_embs = [user_rec_embs_all[user], user_src_embs_all[user]]
        code_embs = [
            rec_code_emb_all[user_rec_code], src_code_emb_all[user_src_code]
        ]
        user_cl_used = [user_rec_embs_all[user], user_src_embs_all[user]]

        return user_embs, code_embs, user_cl_used

    def repeat_feat(self, feat_list: List[torch.Tensor], repeat_num: int):
        return [
            torch.repeat_interleave(feat, repeat_num, dim=0)
            for feat in feat_list
        ]

    def forward(self, user: torch.Tensor, rec_his: torch.Tensor,
                src_session_his: torch.Tensor, true_src: torch.Tensor,
                items_emb: torch.Tensor):
        num_items = items_emb.size(1)

        user_feats = []
        user_embs, code_embs, user_cl_used = self.get_user_emb(user)

        rec_his_emb = self.session_emb.get_item_emb(rec_his)
        rec_his_mask = torch.where(rec_his == 0, 1, 0).bool()
        rec_his_emb = self.rec_his_transformer_layer(
            src=rec_his_emb, src_key_padding_mask=rec_his_mask)

        user_rec_code_emb = code_embs[0]
        rec_code_mask = torch.zeros(
            (user_rec_code_emb.shape[0],
             user_rec_code_emb.shape[1])).bool().to(self.device)

        rec_code_his_cl_used = [
            rec_his_emb, rec_his_mask, user_rec_code_emb, rec_code_mask
        ]

        rec_his_emb = self.rec_code_fusion(
            tgt=rec_his_emb,
            memory=user_rec_code_emb,
            tgt_key_padding_mask=rec_his_mask,
            memory_key_padding_mask=rec_code_mask)

        rec_his_emb_repeat, rec_his_mask_repeat = self.repeat_feat(
            [rec_his_emb, rec_his_mask], num_items)

        src_his_emb = torch.zeros(
            (src_session_his.shape[0], src_session_his.shape[1],
             rec_his_emb.shape[-1])).to(self.device)
        src_his_true = src_session_his[true_src]
        src_his_emb_true = self.src_feat_process(
            self.session_emb(src_his_true))

        src_his_pseudo = src_session_his[~true_src]
        src_his_emb_pseudo = self.session_emb.get_item_emb(src_his_pseudo)

        src_his_emb[true_src] = src_his_emb_true
        src_his_emb[~true_src] = src_his_emb_pseudo

        src_his_mask = torch.where(src_session_his == 0, 1, 0).bool()
        src_his_emb = self.src_his_transformer_layer(
            src=src_his_emb, src_key_padding_mask=src_his_mask)

        user_src_code_emb = code_embs[1]
        src_code_mask = torch.zeros(
            (user_src_code_emb.shape[0],
             user_src_code_emb.shape[1])).bool().to(self.device)

        src_code_his_cl_used = [
            src_his_emb, src_his_mask, user_src_code_emb, src_code_mask
        ]

        src_his_emb = self.src_code_fusion(
            tgt=src_his_emb,
            memory=user_src_code_emb,
            tgt_key_padding_mask=src_his_mask,
            memory_key_padding_mask=src_code_mask)

        src_his_emb_repeat, src_his_mask_repeat = self.repeat_feat(
            [src_his_emb, src_his_mask], num_items)

        items_emb = items_emb.reshape(-1, items_emb.size(-1))

        rec_fusion = self.rec_his_attn_pooling(rec_his_emb_repeat, items_emb,
                                               rec_his_mask_repeat)
        src_fusion = self.src_his_attn_pooling(src_his_emb_repeat, items_emb,
                                               src_his_mask_repeat)

        user_feats.append(rec_fusion)
        user_feats.append(src_fusion)

        user_feats.extend(self.repeat_feat(user_embs, num_items))

        return user_feats, user_cl_used, rec_code_his_cl_used, src_code_his_cl_used

    def inter_pred(self, user_feats, item_emb):
        user_feats = torch.cat(user_feats, dim=-1)

        item_emb = item_emb.reshape(-1, item_emb.size(-1))

        output = torch.cat([user_feats, item_emb], dim=-1)

        return self.rec_fc_layer(output)

    def rec_loss(self, inputs):
        user, rec_his, src_session_his, true_src, pos_item, neg_items = inputs[
            'user'], inputs['rec_his'], inputs['src_session_his'], inputs[
                'true_src'], inputs['item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_emb.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, user_cl_used, rec_code_his_cl_used, src_code_his_cl_used = self.forward(
            user, rec_his, src_session_his, true_src, items_emb)

        logits = self.inter_pred(user_feats, items_emb).reshape(
            (batch_size, -1))

        total_loss = self.loss_fn(logits)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.user_cl_weight > 0:
            user_rec_emb, user_src_emb = user_cl_used
            user_cl_loss = self.user_cl(user_rec_emb, user_src_emb)
            loss_dict['user_cl_loss'] = user_cl_loss.clone()

            total_loss += self.user_cl_weight * user_cl_loss

        if self.code_his_cl_weight > 0:
            rec_his_emb, rec_his_mask, user_rec_code_emb, rec_code_mask = rec_code_his_cl_used
            rec_code_his_cl_loss = self.rec_code_his_cl(
                rec_his_emb, rec_his_mask, user_rec_code_emb, rec_code_mask)
            loss_dict['rec_code_cl'] = rec_code_his_cl_loss.clone()

            total_loss += self.code_his_cl_weight * rec_code_his_cl_loss

            src_his_emb, src_his_mask, user_src_code_emb, src_code_mask = src_code_his_cl_used
            src_code_his_cl_loss = self.src_code_his_cl(
                src_his_emb, src_his_mask, user_src_code_emb, src_code_mask)
            loss_dict['src_code_cl'] = src_code_his_cl_loss.clone()

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def rec_predict(self, inputs):
        user, rec_his, src_session_his,true_src, pos_item, neg_items \
            = inputs['user'], inputs['rec_his'], inputs['src_session_his'], inputs[
                'true_src'],inputs['item'], inputs['neg_items']

        items = torch.cat([neg_items, pos_item.unsqueeze(1)], dim=1)
        items_emb = self.session_emb.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, user_cl_used, rec_code_his_cl_used, src_code_his_cl_used = self.forward(
            user, rec_his, src_session_his, true_src, items_emb)

        all_item_score = self.inter_pred(user_feats, items_emb).reshape(
            (batch_size, -1))

        all_item_ids = items
        sort_score, sort_indices = all_item_score.sort(dim=-1, descending=True)
        sort_item_ids = all_item_ids.gather(1, sort_indices)

        results = (sort_item_ids == pos_item.unsqueeze(1)).long()

        return results


class Target_Attention(nn.Module):

    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec


class EmbCL(nn.Module):

    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, emb_1: torch.Tensor, emb_2: torch.Tensor):
        batch_size = emb_1.size(0)
        N = 2 * batch_size

        z = torch.cat([emb_1.squeeze(), emb_2.squeeze()], dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

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


class HisCL(nn.Module):

    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, his_emb: torch.Tensor, his_mask: torch.Tensor,
                code_emb: torch.Tensor, code_mask: torch.Tensor):
        none_zero_his = (~his_mask).sum(dim=1) > 0
        if none_zero_his.sum() <= 0:
            return torch.tensor([0.]).to(his_emb.device)
        else:
            his_emb = his_emb[none_zero_his]
            his_mask = his_mask[none_zero_his]
            code_emb = code_emb[none_zero_his]
            code_mask = code_mask[none_zero_his]

        his_emb = his_emb.masked_fill(his_mask.unsqueeze(2), 0)
        his_pooling = his_emb.sum(dim=1) / (~his_mask).sum(dim=1, keepdim=True)

        code_emb = code_emb.masked_fill(code_mask.unsqueeze(2), 0)
        code_pooling = code_emb.sum(dim=1) / (~code_mask).sum(dim=1,
                                                              keepdim=True)

        batch_size = his_pooling.size(0)
        N = 2 * batch_size

        z = torch.cat([his_pooling.squeeze(), code_pooling.squeeze()], dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

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
