import logging
import os
from typing import Dict

import torch
import torch.nn as nn
from modules import const, utils

from .layers import FullyConnectedLayer


class UserFeatCode(nn.Module):

    def __init__(self, device, rec_index_path, src_index_path):
        super().__init__()
        self.device = device

        self.rec_index_path = rec_index_path
        self.src_index_path = src_index_path

        self.load_code()

    def load_code(self):
        code2id = {'pad': 0}
        code_idx = 1

        rec_index: Dict = utils.load_json(
            os.path.join(const.load_path, self.rec_index_path))
        src_index: Dict = utils.load_json(
            os.path.join(const.load_path, self.src_index_path))
        logging.info("load code index from %s and %s" %
                     (self.rec_index_path, self.src_index_path))
        logging.info("rec code: {} src code: {}".format(
            rec_index['1'], src_index['1']))

        code_len = len(list(rec_index.values())[0])

        user2rec_code = [[0 for _ in range(code_len)]
                         for _ in range(const.user_id_num)]
        user2src_code = [[0 for _ in range(code_len)]
                         for _ in range(const.user_id_num)]
        user_rec_code_pair = []
        user_src_code_pair = []

        for u_id, code_list in rec_index.items():
            code_id_list = []
            for code in code_list:
                if code not in code2id.keys():
                    code2id[code] = code_idx
                    code_idx += 1
                code_id_list.append(code2id[code])
                user_rec_code_pair.append((int(u_id), code2id[code]))
            user2rec_code[int(u_id)] = code_id_list

        for u_id, code_list in src_index.items():
            code_id_list = []
            for code in code_list:
                if code not in code2id.keys():
                    code2id[code] = code_idx
                    code_idx += 1
                code_id_list.append(code2id[code])
                user_src_code_pair.append((int(u_id), code2id[code]))
            user2src_code[int(u_id)] = code_id_list

        self.user_rec_code_pair = user_rec_code_pair
        self.user_src_code_pair = user_src_code_pair

        self.user2rec_code = torch.tensor(user2rec_code,
                                          dtype=torch.long,
                                          device=self.device)
        self.user2src_code = torch.tensor(user2src_code,
                                          dtype=torch.long,
                                          device=self.device)

        self.code2id = code2id

        self.code_embedding = nn.Embedding(num_embeddings=len(code2id),
                                           embedding_dim=const.final_emb_size,
                                           padding_idx=0)
        nn.init.xavier_normal_(self.code_embedding.weight.data)
        self.code_embedding.weight.data[0] = 0

    def get_all_user_code_emb(self):
        return self.code_embedding.weight


class UserFeat(nn.Module):

    def __init__(self, user_vocab, device, dropout=0.1):
        super().__init__()
        self.user_vocab = user_vocab
        self.device = device
        self.map_vocab = self.build_map_vocab()

        self.use_mlp = False

        self.attr_ls = const.user_feature_list
        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'{attr}_emb',
                nn.Embedding(num_embeddings=getattr(const, f'{attr}_num'),
                             embedding_dim=getattr(const, f'{attr}_dim')))
            nn.init.xavier_normal_(getattr(self, f'{attr}_emb').weight.data)
            self.size += getattr(const, f'{attr}_dim')

        if self.size != const.final_emb_size:
            self.user_trans = FullyConnectedLayer(
                input_size=self.size,
                hidden_unit=[const.final_emb_size],
                activation='relu',
                dropout=dropout)
            utils.init_module_weights(self.user_trans)
            self.use_mlp = True
        else:
            self.user_trans = nn.Identity()

        self.size = const.final_emb_size

    def build_map_vocab(self):
        user_map_vocab = {}
        for user_id, user_info in self.user_vocab.items():
            for attr in const.user_feature_list:
                if attr == 'user_id':
                    continue
                if attr not in user_map_vocab.keys():
                    user_map_vocab[attr] = list(range(const.user_id_num))
                user_map_vocab[attr][user_id] = user_info[attr]

        for attr in user_map_vocab.keys():
            user_map_vocab[attr] = torch.LongTensor(user_map_vocab[attr]).to(
                self.device)
        return user_map_vocab

    def forward(self, sample):
        feats_ls = []
        for attr in self.attr_ls:
            if attr == 'user_id':
                index = sample
            else:
                index = self.map_vocab[attr][sample]

            feats_ls.append(getattr(self, f'{attr}_emb')(index))

        return self.user_trans(torch.cat(feats_ls, dim=-1))

    def get_all_user_emb(self):
        if self.use_mlp:
            all_user_ids = torch.arange(0, const.user_id_num).long().to(
                self.device)
            all_user_embeddings = self.forward(all_user_ids)
            return all_user_embeddings
        else:
            return self.user_id_emb.weight


class ItemFeat(nn.Module):

    def __init__(self, item_vocab, device, query_feat, dropout=0.1):
        super().__init__()
        self.item_vocab = item_vocab
        self.device = device
        self.use_mlp = False

        self.size = 0
        if const.use_item_pretrained_emb:
            item_pretrained_emb: torch.Tensor = torch.load(
                const.item_pretrained_emb, weights_only=True)
            assert len(item_vocab) == (len(item_pretrained_emb) + 1)

            # add pad
            item_pretrained_emb = torch.cat([
                torch.zeros(1, item_pretrained_emb.shape[1]),
                item_pretrained_emb
            ],
                                            dim=0)

            self.item_pretrained_emb = nn.Embedding.from_pretrained(
                item_pretrained_emb,
                freeze=const.freeze_item_pretrained_emb,
                padding_idx=0,
            )
            logging.info("use item pretrained emb: {} shape: {}".format(
                const.item_pretrained_emb,
                self.item_pretrained_emb.weight.data.shape))

            if item_pretrained_emb.shape[1] != const.final_emb_size:
                self.item_pretrained_trans = FullyConnectedLayer(
                    input_size=self.item_pretrained_emb.weight.data.shape[1],
                    hidden_unit=[512, 256, 128, const.final_emb_size],
                    activation='relu',
                    dropout=dropout)
                self.use_mlp = True
            elif const.freeze_item_pretrained_emb:
                self.item_pretrained_trans = FullyConnectedLayer(
                    input_size=self.item_pretrained_emb.weight.data.shape[1],
                    hidden_unit=[
                        const.final_emb_size * 2, const.final_emb_size
                    ],
                    activation='relu',
                    dropout=dropout)
                self.use_mlp = True
            else:
                self.item_pretrained_trans = nn.Identity()

            utils.init_module_weights(self.item_pretrained_trans)

            self.size += const.final_emb_size

        if (not const.use_item_pretrained_emb):

            self.map_vocab = self.build_map_vocab()

            self.attr_ls = const.item_feature_list

            id_size = 0
            for attr in self.attr_ls:
                if attr in const.item_text_feature:
                    setattr(self, f'{attr}_emb', query_feat)
                    id_size += query_feat.size
                else:
                    setattr(
                        self, f'{attr}_emb',
                        nn.Embedding(
                            num_embeddings=getattr(const, f'{attr}_num'),
                            embedding_dim=getattr(const, f'{attr}_dim'),
                            padding_idx=0))
                    nn.init.xavier_normal_(
                        getattr(self, f'{attr}_emb').weight.data)
                    getattr(self, f'{attr}_emb').weight.data[0, :] = 0
                    id_size += getattr(const, f'{attr}_dim')

            if id_size != const.final_emb_size:
                self.item_id_trans = FullyConnectedLayer(
                    input_size=id_size,
                    hidden_unit=[const.final_emb_size],
                    activation='relu',
                    dropout=dropout)
                utils.init_module_weights(self.item_id_trans)
                self.use_mlp = True
            else:
                self.item_id_trans = nn.Identity()

            self.size += const.final_emb_size

    def build_map_vocab(self):
        item_map_vocab = {}
        max_item_id = -1
        for item_id, item_info in self.item_vocab.items():
            max_item_id = max(item_id, max_item_id)
            for attr in const.item_feature_list:
                if attr == 'item_id':
                    continue
                if attr not in item_map_vocab.keys():
                    item_map_vocab[attr] = list(range(len(self.item_vocab)))

                if attr in const.item_text_feature:
                    item_text_ids = sum(
                        [item_info[text_key] for text_key in attr.split('-')],
                        [])
                    item_map_vocab[attr][item_id] = utils.get_pad_sentence(
                        item_text_ids,
                        max_len=const.max_item_text_len,
                        pad_token=0)
                else:
                    item_map_vocab[attr][item_id] = item_info[attr]
        assert max_item_id == len(self.item_vocab) - 1

        for attr in item_map_vocab.keys():
            item_map_vocab[attr] = torch.LongTensor(item_map_vocab[attr]).to(
                self.device)
        return item_map_vocab

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        result_emb = torch.zeros((new_sample.shape[0], const.final_emb_size),
                                 device=sample.device)
        sub_mask = new_sample != 0
        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]

            if const.use_item_pretrained_emb:
                sub_sample_pretrained_emb = self.item_pretrained_trans(
                    self.item_pretrained_emb(sub_sample))

            if (not const.use_item_pretrained_emb):
                feats_ls = []
                for attr in self.attr_ls:
                    if attr == 'item_id':
                        index = sub_sample
                    else:
                        index = self.map_vocab[attr][sub_sample]

                    feats_ls.append(getattr(self, f'{attr}_emb')(index))

                sub_sample_id_emb = self.item_id_trans(
                    torch.cat(feats_ls, dim=-1))

            if const.use_item_pretrained_emb:
                sub_sample_emb = sub_sample_pretrained_emb
            else:
                sub_sample_emb = sub_sample_id_emb

            result_emb[sub_mask] = sub_sample_emb
        result_emb = result_emb.reshape((*sample.shape, const.final_emb_size))

        return result_emb


class QueryEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=const.word_id_num,
                                           embedding_dim=const.word_id_dim,
                                           padding_idx=0)
        nn.init.xavier_normal_(self.word_embedding.weight.data)

    def forward(self, seqs):
        seqs_mask = (seqs == 0)
        seqs_emb = self.word_embedding(seqs)

        output = seqs_emb

        # mean pooling
        seqs_len = (~seqs_mask).sum(1, keepdim=True)
        output = output.masked_fill(seqs_mask.unsqueeze(2), 0)
        sum_emb = output.sum(dim=1)
        mean_emb = sum_emb / seqs_len

        mean_emb = mean_emb.masked_fill(seqs_len == 0, 0)

        return mean_emb.squeeze()


class QueryFeat(nn.Module):

    def __init__(self, device: torch.device, query_vocab, dropout=0.1):
        super().__init__()
        self.device = device
        self.query_vocab = query_vocab

        if const.use_query_pretrained_emb:
            query_emb: torch.Tensor = torch.load(const.query_pretrained_emb,
                                                 weights_only=True)
            assert len(query_vocab) == (len(query_emb) + 1)

            query_emb = torch.cat(
                [torch.zeros((1, query_emb.shape[1])), query_emb], dim=0)

            self.query_emb = nn.Embedding.from_pretrained(
                query_emb,
                freeze=const.freeze_query_pretrained_emb,
                padding_idx=0)
            self.size = self.query_emb.weight.data.shape[1]
            logging.info("use query pretrained emb: {} shape: {}".format(
                const.query_pretrained_emb, self.query_emb.weight.data.shape))

            if query_emb.shape[1] != const.final_emb_size:
                self.query_trans = FullyConnectedLayer(
                    input_size=self.query_emb.weight.data.shape[1],
                    hidden_unit=[512, 256, 128, const.final_emb_size],
                    activation='relu',
                    dropout=dropout)
            elif const.freeze_query_pretrained_emb:
                self.query_trans = FullyConnectedLayer(
                    input_size=self.query_emb.weight.data.shape[1],
                    hidden_unit=[
                        const.final_emb_size * 2, const.final_emb_size
                    ],
                    activation='relu',
                    dropout=dropout)
            else:
                self.query_trans = nn.Identity()

        else:
            self.query_encoder = QueryEncoder()
            self.size = const.word_id_dim

            if self.size != const.final_emb_size:
                self.query_trans = FullyConnectedLayer(
                    input_size=self.size,
                    hidden_unit=[const.final_emb_size],
                    activation='relu',
                    dropout=dropout)
            else:
                self.query_trans = nn.Identity()

        self.size = const.final_emb_size

        utils.init_module_weights(self.query_trans)

    def forward(self, sample: torch.Tensor):
        if const.use_query_pretrained_emb:
            query_emb: torch.Tensor = self.query_emb(sample)

            return self.query_trans(query_emb)
        else:
            query_emb: torch.Tensor = self.query_encoder(
                sample.reshape((-1, sample.shape[-1])))
            query_emb = query_emb.reshape((*sample.shape[:-1], -1))

            return self.query_trans(query_emb)


class SrcSessionFeat(nn.Module):

    def __init__(self,
                 query_feat,
                 item_feat,
                 session_vocab,
                 device,
                 user_feat=None):
        super().__init__()
        self.query_feat = query_feat
        self.item_feat = item_feat
        self.user_feat = user_feat

        self.session_vocab = session_vocab
        self.device = device
        self.map_vocab = self.build_map_vocab()

    def build_map_vocab(self):
        session_keys = list(self.session_vocab.keys())
        session_map_vocab = {
            "query_id": list(range(max(session_keys) + 1)),
            "keyword": list(range(max(session_keys) + 1)),
            "pos_items": list(range(max(session_keys) + 1))
        }

        for session_id, session_info in self.session_vocab.items():
            if 'query_id' in session_info.keys():
                session_map_vocab['query_id'][session_id] = session_info[
                    'query_id']

            session_map_vocab['keyword'][session_id] = utils.get_pad_sentence(
                session_info['keyword'],
                max_len=const.max_query_word_len,
                pad_token=0)
            session_map_vocab['pos_items'][session_id] = utils.get_pad_seqs(
                session_info['pos_items'], max_len=const.max_session_item_len)

        session_map_vocab['keyword'][0] = [0] * const.max_query_word_len
        session_map_vocab['pos_items'][0] = [0] * const.max_session_item_len
        for attr in session_map_vocab.keys():
            session_map_vocab[attr] = torch.LongTensor(
                session_map_vocab[attr]).to(self.device)
        return session_map_vocab

    def get_user_emb(self, sample, all_user_embs=None):
        if all_user_embs is not None:
            return all_user_embs[sample]
        else:
            if self.user_feat is None:
                raise NotImplementedError
            return self.user_feat(sample)

    def get_item_emb(self, sample, all_item_embs=None, **kwargs):
        if all_item_embs is not None:
            return all_item_embs[sample]
        else:
            if self.item_feat is None:
                raise NotImplementedError
            return self.item_feat(sample, **kwargs)

    def get_query_emb(self, sample):
        return self.query_feat(sample)

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        sub_mask = new_sample != 0

        result_query_emb = torch.zeros(
            (new_sample.shape[0], const.final_emb_size), device=sample.device)
        result_item_emb = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len,
             const.final_emb_size),
            device=sample.device)
        result_item_mask = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len),
            device=sample.device).bool()

        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]
            sub_click_item_ls = self.map_vocab['pos_items'][sub_sample]

            if const.use_query_pretrained_emb:
                sub_query_id = self.map_vocab['query_id'][sub_sample]
            else:
                sub_query_id = self.map_vocab['keyword'][sub_sample]
            sub_query_emb = self.get_query_emb(sub_query_id)

            sub_click_item_mask = torch.where(sub_click_item_ls == 0, 0,
                                              1).bool()
            sub_click_item_emb = self.get_item_emb(sub_click_item_ls)

            result_query_emb[sub_mask] = sub_query_emb
            result_item_emb[sub_mask] = sub_click_item_emb
            result_item_mask[sub_mask] = sub_click_item_mask

        result_query_emb = result_query_emb.reshape(
            (*sample.shape, const.final_emb_size))
        result_item_emb = result_item_emb.reshape(
            (*sample.shape, const.max_session_item_len, const.final_emb_size))
        result_item_mask = result_item_mask.reshape(
            (*sample.shape, const.max_session_item_len))

        return [result_query_emb, result_item_emb, result_item_mask]
