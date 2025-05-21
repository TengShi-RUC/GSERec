import os


def init_setting_Amazon_CDs():
    global load_path, user_vocab, item_vocab, query_vocab, session_vocab, \
        rec_train, rec_val, rec_test

    load_path = "../data/Amazon_CDs"
    user_vocab = os.path.join(load_path, 'vocab/user_vocab.pkl')
    item_vocab = os.path.join(load_path, 'vocab/item_vocab.pkl')
    query_vocab = os.path.join(load_path, 'vocab/query_vocab.pkl')
    session_vocab = os.path.join(load_path, 'vocab/src_session_vocab.pkl')

    rec_train = os.path.join(load_path, 'dataset/rec_train.pkl')
    rec_val = os.path.join(load_path, 'dataset/rec_val.pkl')
    rec_test = os.path.join(load_path, 'dataset/rec_test.pkl')

    global item_id_num, item_id_dim, item_feature_list, item_text_feature, item_mask_token

    item_feature_list = ['item_id']
    item_text_feature = []

    item_id_num = 62960 + 1
    item_mask_token = 62960
    item_id_dim = 64

    global user_id_num, user_id_dim, user_feature_list
    user_feature_list = ['user_id']
    user_id_num = 75258
    user_id_dim = 64

    global word_id_num, word_id_dim
    word_id_num = 726
    word_id_dim = 64

    global final_emb_size
    final_emb_size = 64

    global max_rec_his_len, max_src_session_his_len, max_session_item_len
    max_rec_his_len = 20
    max_src_session_his_len = 20
    max_session_item_len = 5

    global  item_pretrained_emb, query_pretrained_emb, \
        freeze_item_pretrained_emb,freeze_query_pretrained_emb,\
        use_item_pretrained_emb,use_query_pretrained_emb

    use_item_pretrained_emb = False
    item_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_item.pt')
    freeze_item_pretrained_emb = True

    use_query_pretrained_emb = False
    query_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_query.pt')
    freeze_query_pretrained_emb = True

    global max_query_word_len, max_item_text_len
    max_query_word_len = 64
    max_item_text_len = 256


def init_setting_Amazon_Electronics():
    global load_path, user_vocab, item_vocab, query_vocab, session_vocab, \
        rec_train, rec_val, rec_test

    load_path = "../data/Amazon_Electronics"
    user_vocab = os.path.join(load_path, 'vocab/user_vocab.pkl')
    item_vocab = os.path.join(load_path, 'vocab/item_vocab.pkl')
    query_vocab = os.path.join(load_path, 'vocab/query_vocab.pkl')
    session_vocab = os.path.join(load_path, 'vocab/src_session_vocab.pkl')

    rec_train = os.path.join(load_path, 'dataset/rec_train.pkl')
    rec_val = os.path.join(load_path, 'dataset/rec_val.pkl')
    rec_test = os.path.join(load_path, 'dataset/rec_test.pkl')

    global item_id_num, item_id_dim, item_feature_list, item_text_feature, item_mask_token

    item_feature_list = ['item_id']
    item_text_feature = []

    item_id_num = 62883 + 1
    item_mask_token = 62883
    item_id_dim = 64

    global user_id_num, user_id_dim, user_feature_list
    user_feature_list = ['user_id']
    user_id_num = 192403
    user_id_dim = 64

    global word_id_num, word_id_dim
    word_id_num = 719
    word_id_dim = 64

    global final_emb_size
    final_emb_size = 64

    global max_rec_his_len, max_src_session_his_len, max_session_item_len
    max_rec_his_len = 10
    max_src_session_his_len = 10
    max_session_item_len = 1

    global  item_pretrained_emb, query_pretrained_emb, \
        freeze_item_pretrained_emb,freeze_query_pretrained_emb,\
        use_item_pretrained_emb,use_query_pretrained_emb

    use_item_pretrained_emb = False
    item_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_item.pt')
    freeze_item_pretrained_emb = True

    use_query_pretrained_emb = False
    query_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_query.pt')
    freeze_query_pretrained_emb = True

    global max_query_word_len, max_item_text_len
    max_query_word_len = 64
    max_item_text_len = 256


def init_setting_Qilin():
    global load_path, user_vocab, item_vocab, query_vocab, session_vocab, \
        rec_train, rec_val, rec_test

    load_path = "../data/Qilin"
    user_vocab = os.path.join(load_path, 'vocab/user_vocab.pkl')
    item_vocab = os.path.join(load_path, 'vocab/item_vocab.pkl')
    query_vocab = os.path.join(load_path, 'vocab/query_vocab.pkl')
    session_vocab = os.path.join(load_path, 'vocab/src_session_vocab.pkl')

    rec_train = os.path.join(load_path, 'dataset/rec_train.pkl')
    rec_val = os.path.join(load_path, 'dataset/rec_val.pkl')
    rec_test = os.path.join(load_path, 'dataset/rec_test.pkl')

    global item_id_num, item_id_dim, item_feature_list, item_text_feature

    item_feature_list = ['item_id']
    item_text_feature = []

    item_id_num = 402928
    item_id_dim = 32

    global user_id_num, user_id_dim, user_feature_list
    user_feature_list = ['user_id']
    user_id_num = 12389
    user_id_dim = 32

    global word_id_num, word_id_dim
    word_id_num = 537213
    word_id_dim = 32

    global final_emb_size
    final_emb_size = 32

    global max_rec_his_len, max_src_session_his_len, max_session_item_len
    max_rec_his_len = 20
    max_src_session_his_len = 20
    max_session_item_len = 5


    global  item_pretrained_emb, query_pretrained_emb, \
        freeze_item_pretrained_emb,freeze_query_pretrained_emb,\
        use_item_pretrained_emb,use_query_pretrained_emb

    use_item_pretrained_emb = True
    item_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_item.pt')
    freeze_item_pretrained_emb = True

    use_query_pretrained_emb = True
    query_pretrained_emb = os.path.join(load_path, 'emb/bge-m3_query.pt')
    freeze_query_pretrained_emb = True
    # Due to the sparsity of the Qilin dataset, we use pretrained embeddings for both items and queries across all models (including baselines and GSERec).

    global max_query_word_len, max_item_text_len
    max_query_word_len = 64
    max_item_text_len = 256
