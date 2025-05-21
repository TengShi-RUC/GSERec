import datetime
import json
import logging
import math
import os
import pickle
import random
from typing import Any, Dict, List, NoReturn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modules import const


def printSetting():

    for attr in const.item_feature_list:
        if attr in const.item_text_feature:
            logging.info("using item {}".format(attr))
            continue
        logging.info("{} num:{} dim:{}".format(attr,
                                               getattr(const, f"{attr}_num"),
                                               getattr(const, f"{attr}_dim")))

    for attr in const.user_feature_list:
        logging.info("{} num:{} dim:{}".format(attr,
                                               getattr(const, f"{attr}_num"),
                                               getattr(const, f"{attr}_dim")))

    logging.info("{} num:{} dim:{}".format('word_id', const.word_id_num,
                                           const.word_id_dim))

    logging.info("final_emb_size:{}".format(const.final_emb_size))

    logging.info("max_rec_his_len:{}".format(const.max_rec_his_len))
    logging.info("max_src_session_his_len:{}".format(
        const.max_src_session_his_len))
    logging.info("max_session_item_len:{}".format(const.max_session_item_len))

    logging.info("use_item_pretrained_emb:{}".format(
        const.use_item_pretrained_emb))
    logging.info("item_pretrained_emb:{}".format(const.item_pretrained_emb))
    logging.info("freeze_item_pretrained_emb:{}".format(
        const.freeze_item_pretrained_emb))

    logging.info("use_query_pretrained_emb:{}".format(
        const.use_query_pretrained_emb))
    logging.info("query_pretrained_emb:{}".format(const.query_pretrained_emb))
    logging.info("freeze_query_pretrained_emb:{}".format(
        const.freeze_query_pretrained_emb))

    logging.info("max_query_word_len:{}".format(const.max_query_word_len))
    logging.info("max_item_text_len:{}".format(const.max_item_text_len))


def init_module_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


GLOBAL_SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def load_json(path):
    return json.load(open(path, 'r'))


def load_df(data_path: str):
    if data_path.endswith("json"):
        return pd.read_json(data_path)
    elif data_path.endswith("pkl"):
        return pd.read_pickle(data_path)


def count_variables(model: nn.Module) -> int:
    total_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_p = p.numel()
            total_parameters += num_p

    return total_parameters


def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if isinstance(batch[c], torch.Tensor):
            batch[c] = batch[c].to(device)
        elif isinstance(batch[c], List):
            if isinstance(batch[c][0], str):
                continue
            batch[c] = [[p.to(device)
                         for p in k] if isinstance(k, List) else k.to(device)
                        for k in batch[c]]
    return batch


def ndcg_k(topk_results, k):
    ndcg = 0.0
    results = topk_results[:k]
    for i in range(len(results)):
        ndcg += results[i] / math.log(i + 2, 2)
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    results = topk_results[:k]
    if sum(results) > 0:
        hit = 1.0
    return hit


def format_metric(result_dict: Dict[str, Any]) -> str:
    assert type(result_dict) == dict
    if 'rec' in result_dict.keys():
        return {
            "rec": format_metric(result_dict['rec']),
            "src": format_metric(result_dict['src'])
        }

    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name]
            if type(m) is float or type(m) is np.float32 or type(
                    m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def check_dir(file_name: str) -> NoReturn:
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        logging.info('make dirs:{}'.format(dir_path))
        os.makedirs(dir_path)


def non_increasing(lst: list) -> bool:
    return all(x >= y for x, y in zip(lst, lst[1:]))


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_pad_sentence(sentence, max_len, pad_token=0):
    if type(sentence) == str:
        sentence = eval(sentence)
    if type(sentence) == int:
        sentence = [sentence]
    sentence = sentence[:max_len]
    if len(sentence) < max_len:
        sentence += [pad_token] * (max_len - len(sentence))
    return sentence


def get_pad_seqs(seqs, max_len, pad_token=0):
    seqs = seqs[-max_len:]
    if len(seqs) < max_len:
        seqs += [pad_token] * (max_len - len(seqs))
    return seqs


def load_vocabs():
    logging.info('load vocabs')
    return {
        "user_vocab": load_pickle(const.user_vocab),
        "item_vocab": load_pickle(const.item_vocab),
        "query_vocab": load_pickle(const.query_vocab),
        "session_vocab": load_pickle(const.session_vocab)
    }
