import argparse
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer


def load_item_text(args):
    item_vocab_path = os.path.join("../data", args.dataset,
                                   "vocab/item_vocab.pkl")
    print("load item vocab: {}".format(item_vocab_path))

    item_vocab = pickle.load(open(item_vocab_path, "rb"))
    item_text = []
    for item_id in range(1, len(item_vocab)):
        cur_item_text = item_vocab[item_id]
        item_text.append(cur_item_text['text'])

    return item_text


def load_query_text(args):
    query_vocab_path = os.path.join("../data", args.dataset,
                                    "vocab/query_vocab.pkl")
    print("load query vocab: {}".format(query_vocab_path))

    query_vocab = pickle.load(open(query_vocab_path, "rb"))
    query_text = []
    for query_id in range(1, len(query_vocab)):
        cur_query_text = query_vocab[query_id]
        query_text.append(cur_query_text['query'])

    return query_text


def load_text_encoder(args):
    model_path = os.path.join(args.llm_path, args.llm_name)
    print("load encoder: {}".format(model_path))

    model_path = os.path.join(model_path)

    model = SentenceTransformer(model_path)
    print(model)
    return model


def generate_embedding_sentence(args,
                                text_list,
                                model: SentenceTransformer,
                                is_doc=False):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    if is_doc:
        model.max_seq_length = args.max_doc_len
    else:
        model.max_seq_length = args.max_query_len

    batch_size = args.batch_size
    with torch.no_grad():

        pool = model.start_multi_process_pool()
        embeddings = model.encode_multi_process(text_list,
                                                pool=pool,
                                                batch_size=batch_size,
                                                show_progress_bar=True,
                                                normalize_embeddings=True)
        model.stop_multi_process_pool(pool)

        embeddings = torch.from_numpy(embeddings)

        embeddings = embeddings.cpu()
    print('Embeddings shape: ', embeddings.shape)

    save_path = os.path.join("../data", args.dataset, "emb")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if is_doc:
        save_path = os.path.join(save_path, f"{args.llm_name}_item.pt")
    else:
        save_path = os.path.join(save_path, f"{args.llm_name}_query.pt")

    torch.save(embeddings, save_path)
    print("save embedding to: {}".format(save_path))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Qilin')

parser.add_argument('--llm_name', type=str, default='bge-m3')
parser.add_argument('--llm_path', type=str, default='LLMs')

parser.add_argument('--max_doc_len', type=int, default=256)
parser.add_argument('--max_query_len', type=int, default=64)

parser.add_argument('--batch_size', type=int, default=128)

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0,1 python emb/get_item_query_emb.py
    args = parser.parse_args()
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    item_text_list = load_item_text(args)
    query_text_list = load_query_text(args)

    model = load_text_encoder(args)
    generate_embedding_sentence(args, item_text_list, model, is_doc=True)
    generate_embedding_sentence(args, query_text_list, model, is_doc=False)
