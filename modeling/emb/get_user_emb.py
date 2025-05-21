import argparse
import json
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer


def load_user_text(args):
    user_vocab_path = os.path.join("../data", args.dataset,
                                   "vocab/user_vocab.pkl")
    print("load user vocab: {}".format(user_vocab_path))
    user_vocab = pickle.load(open(user_vocab_path, "rb"))

    user_rec_prefers = ['' for _ in range(len(user_vocab))]
    user_src_prefers = ['' for _ in range(len(user_vocab))]

    user_rec_prefer_path = os.path.join('../data', args.dataset,
                                        args.llm_rec_reason)
    print("load user rec prefers: {}".format(user_rec_prefer_path))

    user_src_prefer_path = os.path.join('../data', args.dataset,
                                        args.llm_src_reason)
    print("load user src prefers: {}".format(user_src_prefer_path))

    if os.path.isdir(user_rec_prefer_path):
        user_rec_prefer_file = []
        for file in os.listdir(user_rec_prefer_path):
            print("load user rec prefers: {}".format(
                os.path.join(user_rec_prefer_path, file)))
            user_rec_prefer_file.extend(
                json.load(open(os.path.join(user_rec_prefer_path, file), "r")))
    else:
        print("user rec prefers: {}".format(user_rec_prefer_path))
        user_rec_prefer_file = json.load(open(user_rec_prefer_path, "r"))

    if os.path.isdir(user_src_prefer_path):
        user_src_prefer_file = []
        for file in os.listdir(user_src_prefer_path):
            print("load user src prefers: {}".format(
                os.path.join(user_src_prefer_path, file)))
            user_src_prefer_file.extend(
                json.load(open(os.path.join(user_src_prefer_path, file), "r")))
    else:
        print("user src prefers: {}".format(user_src_prefer_path))
        user_src_prefer_file = json.load(open(user_src_prefer_path, "r"))

    for idx, data in enumerate(user_rec_prefer_file):
        try:
            assert len(data['predict']) != 0
            cur_prefer = data['predict']
        except:
            print("data {} lack reason".format(idx))
            cur_prefer = data['output']

        user_rec_prefers[int(data['user_id'])] = cur_prefer

    for idx, data in enumerate(user_src_prefer_file):
        try:
            assert len(data['predict']) != 0
            cur_prefer = data['predict']
        except:
            print("data {} lack reason".format(idx))
            cur_prefer = data['output']

        user_src_prefers[int(data['user_id'])] = cur_prefer

    for idx in range(len(user_rec_prefers)):
        if user_rec_prefers[idx] == '':
            user_rec_prefers[idx] = user_src_prefers[idx]

    for idx in range(len(user_src_prefers)):
        if user_src_prefers[idx] == '':
            user_src_prefers[idx] = user_rec_prefers[idx]

    return user_rec_prefers, user_src_prefers


def load_text_encoder(args):
    model_path = os.path.join(args.emb_model_path, args.emb_model_name)
    print("load encoder: {}".format(model_path))

    model_path = os.path.join(model_path)

    model = SentenceTransformer(model_path)
    print(model)
    return model


def generate_embedding_sentence(args,
                                text_list,
                                model: SentenceTransformer,
                                domain='rec'):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

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

    if domain == 'rec':
        llm_name = args.llm_rec_reason.split('/')[-2]
    elif domain == 'src':
        llm_name = args.llm_src_reason.split('/')[-2]
    else:
        raise NotImplementedError
    save_path = os.path.join("../data", args.dataset, "emb", llm_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path,
                             f"{args.emb_model_name}_user_{domain}.pt")

    torch.save(embeddings, save_path)
    print("save embedding to: {}".format(save_path))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Qilin')

parser.add_argument('--emb_model_name', type=str, default='bge-m3')
parser.add_argument('--emb_model_path',
                    type=str,
                    default='LLMs')

parser.add_argument(
    '--llm_rec_reason',
    type=str,
    default="generate/DeepSeek-R1-Distill-Qwen-7B_rec-20/")
parser.add_argument(
    '--llm_src_reason',
    type=str,
    default="generate/DeepSeek-R1-Distill-Qwen-7B_src-20/")

parser.add_argument('--batch_size', type=int, default=8)

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0,1 python emb/get_user_emb.py
    args = parser.parse_args()
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    user_rec_prefers, user_src_prefers = load_user_text(args)

    model = load_text_encoder(args)
    generate_embedding_sentence(args, user_rec_prefers, model, domain='rec')
    generate_embedding_sentence(args, user_src_prefers, model, domain='src')
