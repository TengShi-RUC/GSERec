import os
import pickle

import jieba
import pandas as pd
from tqdm import tqdm

stopwords = []
with open('../stopwords/baidu_stopwords.txt', 'r') as fp:
    for line in fp:
        stopwords.append(line.strip())
print("stopwords: {}".format(len(stopwords)))

word2id = {'<pad>': 0}
id2word = ['<pad>']


def add_word(word):
    global word2id, id2word
    if word not in word2id.keys():
        word2id[word] = len(word2id)
        id2word.append(word)
    return word2id[word]


query2id = {'<pad>': 0}
id2query = [{"query": "<pad>", "words": ["<pad>"], "words_id": [0]}]


def sentence_tokenize(sentence, is_query=False, max_len=256):
    global query2id, id2query
    words = [x for x in jieba.cut(sentence) if x not in stopwords][:max_len]
    words_id = [add_word(x) for x in words]
    if is_query:
        if sentence not in query2id.keys():
            query2id[sentence] = len(query2id)
            id2query.append({
                "query": sentence,
                "words": words,
                "words_id": words_id
            })
        return words, words_id, query2id[sentence]
    else:
        return words, words_id


item_features: pd.DataFrame = pd.read_pickle('raw_data/item_feat.pkl')
print("item_features: {}".format(item_features.shape))


def tokenize_item_row(row):
    text_words, text_words_id = sentence_tokenize(row['text'])

    results = row.to_dict()
    results['text_words'] = text_words
    results['text_words_id'] = text_words_id

    return results


item_features_encoded_data = []
item_features_bar = tqdm(item_features.iterrows())
for idx, line in item_features_bar:
    item_features_bar.set_description('word_vocab: {}'.format(len(id2word)))
    item_features_encoded_data.append(tokenize_item_row(line))
item_features_encoded = pd.DataFrame(item_features_encoded_data)
print("item_features_encoded: {}".format(item_features_encoded.shape))

src_inter: pd.DataFrame = pd.read_pickle('raw_data/src_inter.pkl')
print("src_inter: {}".format(src_inter.shape))

src_inter = src_inter[src_inter['item_id'].isin(
    item_features['item_id'])].reset_index(drop=True)
print("src_inter: {}".format(src_inter.shape))


def tokenize_src_row(row):
    keyword = row['query']
    words, words_id, query_id = sentence_tokenize(keyword, is_query=True)

    results = row.to_dict()
    results['keyword'] = words_id
    results['query_id'] = query_id

    return results


src_inter_encoded_data = []
src_inter_bar = tqdm(src_inter.iterrows())
for idx, line in src_inter_bar:
    src_inter_bar.set_description("word_vocab: {} query_vocab: {}".format(
        len(id2word), len(id2query)))
    src_inter_encoded_data.append(tokenize_src_row(line))
src_inter_encoded = pd.DataFrame(src_inter_encoded_data)
print("src_inter_encoded: {}".format(src_inter_encoded.shape))

if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

if not os.path.exists('vocab'):
    os.makedirs('vocab')

item_features_encoded.to_pickle('raw_data/item_feat_encoded.pkl')

src_inter_encoded.to_pickle('raw_data/src_inter_encoded.pkl')

pickle.dump(id2query, open("vocab/query_vocab.pkl", 'wb'))
pickle.dump(id2word, open("vocab/word_vocab.pkl", 'wb'))

print('word_vocab: {}'.format(len(id2word)))
print('query_vocab: {}'.format(len(id2query)))
