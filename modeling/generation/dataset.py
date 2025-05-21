import sys

sys.path.append('.')

import os
import pickle

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from generation.prompt import *


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = args.dataset
        self.max_doc_len = args.max_doc_len

        self.base_path = os.path.join('../data/', args.dataset)

        self.item_vocab = pickle.load(
            open(os.path.join(self.base_path, 'vocab/item_vocab.pkl'), 'rb'))

        self.user_vocab = pickle.load(
            open(os.path.join(self.base_path, 'vocab/user_vocab.pkl'), 'rb'))

    def get_item_text(self, item_id):
        item_text = self.item_vocab[item_id]['text']
        item_text = self.tokenizer.decode(
            self.tokenizer(item_text,
                           max_length=self.max_doc_len,
                           truncation=True)['input_ids'],
            skip_special_tokens=True)

        return item_text

    def get_query_text(self, query_id):
        query = self.query_vocab[query_id]['query']
        query = self.tokenizer.decode(
            self.tokenizer(query,
                           max_length=self.max_query_len,
                           truncation=True)['input_ids'],
            skip_special_tokens=True)
        return query

    def get_src_train_inter(self):

        src_train: pd.DataFrame = pd.read_pickle(
            f'../data/{self.dataset}/dataset/src_train.pkl')

        if 'timestamp' in src_train.columns:
            time_key = 'timestamp'
        elif 'ts' in src_train.columns:
            time_key = 'ts'
        else:
            raise ValueError('No timestamp or ts in src_train')
        src_train = src_train.sort_values(
            by=['user_id', time_key]).reset_index(drop=True)

        src_train_session = src_train.groupby(
            by=['user_id', 'search_session_id']).agg(
                item_list=("item_id", list),
                query_list=("query_id", list),
                time_list=(time_key, list),
            ).reset_index()

        src_train_session_group = src_train_session.groupby(
            by=['user_id']).agg(session_ids=('search_session_id', list),
                                session_items=("item_list", list),
                                session_querys=("query_list", list),
                                session_times=("time_list",
                                               list)).reset_index()
        src_train_session_group['num_src_train'] = src_train_session_group[
            'session_items'].apply(lambda x: sum([len(t) for t in x]))
        print(src_train_session_group['num_src_train'].describe())

        return src_train_session_group

    def get_rec_train_inter(self):
        rec_train: pd.DataFrame = pd.read_pickle(
            f'../data/{self.dataset}/dataset/rec_train.pkl')
        if 'timestamp' in rec_train.columns:
            time_key = 'timestamp'
        elif 'ts' in rec_train.columns:
            time_key = 'ts'
        else:
            raise ValueError('No timestamp or ts in src_train')
        rec_train = rec_train.sort_values(
            by=['user_id', time_key]).reset_index(drop=True)

        rec_train_group = rec_train.groupby(by=['user_id']).agg(
            item_list=("item_id", list),
            time_list=(time_key, list),
        ).reset_index()

        rec_train_group['num_rec_train'] = rec_train_group['item_list'].apply(
            len)
        print(rec_train_group['num_rec_train'].describe())

        return rec_train_group


class SrcReasonDataset(BaseDataset):

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.max_src_session_his_len = args.max_src_session_his_len
        self.max_session_item_len = args.max_session_item_len
        self.max_query_len = args.max_query_len

        if self.dataset in [
                'Kuaishou_v2_leave_v1', 'Qilin', 'Qilin_v1', 'Qilin_v2'
        ]:
            self.src_session_prompt = src_session_prompt_zh
            self.src_reason_prompt = src_reason_prompt_zh
            self.src_reason_instruction = src_reason_instruction_zh

        elif self.dataset in [
                'Amazon_CDs', 'Amazon_CDs_v1', 'Amazon_Cell', 'Amazon_Cell_v1',
                'Amazon_Electronics', 'Amazon_Electronics_v1', 'Amazon_Kindle',
                'Amazon_Kindle_v1'
        ]:
            self.src_session_prompt = src_session_prompt_en
            self.src_reason_prompt = src_reason_prompt_en
            self.src_reason_instruction = src_reason_instruction_en
        else:
            raise NotImplementedError

        self.query_vocab = pickle.load(
            open(os.path.join(self.base_path, 'vocab/query_vocab.pkl'), 'rb'))

        src_train_session_group = self.get_src_train_inter()
        self.src_train_session_vocab = src_train_session_group.set_index(
            'user_id', drop=True).to_dict('index')

        user_ids = list(self.src_train_session_vocab.keys())
        print("src_train_session_vocab len: {}".format(
            len(self.src_train_session_vocab)))

        self.begin_idx = max(0, args.begin_idx)
        self.end_idx = min(len(user_ids), args.end_idx)

        self.sub_user_ids = user_ids[self.begin_idx:self.end_idx]

    def __getitem__(self, index):
        cur_user = self.sub_user_ids[index]
        cur_user_vocab = self.src_train_session_vocab[cur_user]
        session_items = [
            x[:self.max_session_item_len] for x in
            cur_user_vocab['session_items'][-self.max_src_session_his_len:]
        ]
        session_querys = [
            x[0] for x in cur_user_vocab['session_querys']
            [-self.max_src_session_his_len:]
        ]

        all_prompts = []
        for query, items in zip(session_querys, session_items):
            query_text = self.get_query_text(query)
            item_texts = [self.get_item_text(x) for x in items]
            all_prompts.append(
                self.src_session_prompt.format(**{
                    "query": query_text,
                    "items": ". ".join(item_texts)
                }))

        return {
            "user_id":
            cur_user,
            "input":
            self.src_reason_instruction + self.src_reason_prompt.format(
                **{"session_prompt": "".join(all_prompts)}),
            "num_train_inter":
            cur_user_vocab['num_src_train']
        }

    def __len__(self):
        return len(self.sub_user_ids)


class RecReasonDataset(BaseDataset):

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.max_rec_his_len = args.max_rec_his_len

        if self.dataset in [
                'Kuaishou_v2_leave_v1', 'Qilin', 'Qilin_v1', 'Qilin_v2'
        ]:
            self.rec_reason_prompt = rec_reason_prompt_zh
            self.rec_reason_instruction = rec_reason_instruction_zh
        elif self.dataset in [
                'Amazon_CDs', 'Amazon_CDs_v1', 'Amazon_Cell', 'Amazon_Cell_v1',
                'Amazon_Electronics', 'Amazon_Electronics_v1', 'Amazon_Kindle',
                'Amazon_Kindle_v1'
        ]:
            self.rec_reason_prompt = rec_reason_prompt_en
            self.rec_reason_instruction = rec_reason_instruction_en
        else:
            raise NotImplementedError

        rec_train_group = self.get_rec_train_inter()
        self.rec_train_vocab = rec_train_group.set_index(
            'user_id', drop=True).to_dict('index')
        user_ids = list(self.rec_train_vocab.keys())

        self.begin_idx = max(0, args.begin_idx)
        self.end_idx = min(len(user_ids), args.end_idx)

        self.sub_user_ids = user_ids[self.begin_idx:self.end_idx]

        print("rec_train_vocab len: {}".format(len(self.rec_train_vocab)))

    def __getitem__(self, index):
        cur_user = self.sub_user_ids[index]
        cur_user_vocab = self.rec_train_vocab[cur_user]

        rec_his_ids = cur_user_vocab['item_list'][-self.max_rec_his_len:]
        rec_his_texts = [self.get_item_text(x) for x in rec_his_ids]

        prompts = self.rec_reason_prompt.format(
            **{"rec_his": "\n".join(rec_his_texts)})

        return {
            "user_id": cur_user,
            "input": self.rec_reason_instruction + prompts,
            "num_train_inter": cur_user_vocab['num_rec_train']
        }

    def __len__(self):
        return len(self.sub_user_ids)
