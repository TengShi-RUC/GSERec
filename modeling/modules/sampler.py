import logging
from typing import List

import pandas as pd
from modules import const, utils


class Sampler(object):

    def __init__(self, data_path, user_vocab) -> None:
        super().__init__()
        self.user_vocab = user_vocab

        if isinstance(data_path, List):
            data_list = []
            for file in data_path:
                logging.info("load data: {}".format(file))
                data_list.append(utils.load_df(file))
            self.data = pd.concat(data_list, axis=0).reset_index(drop=True)
        elif isinstance(data_path, str):
            self.data: pd.DataFrame = utils.load_df(data_path)

    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user = int(line['user_id'])
        feed_dict['user'] = [user]

        item = int(line['item_id'])
        feed_dict['item'] = [item]

        neg_items = line['neg_items']
        feed_dict['neg_items'] = [neg_items]

        rec_his_num = int(line['rec_his'])
        if rec_his_num > 0:
            rec_his = self.get_rec_his(user,
                                       rec_his_num,
                                       max_rec_his_len=const.max_rec_his_len)
        else:
            rec_his = [0] * const.max_rec_his_len
        feed_dict['rec_his'] = [rec_his]

        src_session_his_num = int(line['src_session_his'])
        if src_session_his_num > 0:
            src_session_his = self.get_src_session_his(
                user,
                src_session_his_num,
                max_src_session_his_len=const.max_src_session_his_len)
            true_src = True

        else:
            src_session_his = [0] * const.max_src_session_his_len
            true_src = False

        feed_dict['src_session_his'] = [src_session_his]
        feed_dict['true_src'] = [true_src]

        return feed_dict

    def get_rec_his(self, user, rec_his_num, max_rec_his_len):
        rec_his = self.user_vocab[user]['rec_his'][:rec_his_num][
            -max_rec_his_len:]
        rec_his = utils.get_pad_seqs(rec_his, max_len=max_rec_his_len)

        return rec_his

    def get_src_session_his(self, user, src_session_his_num,
                            max_src_session_his_len):
        src_session_his = self.user_vocab[user][
            'src_session_his'][:src_session_his_num][-max_src_session_his_len:]
        src_session_his = utils.get_pad_seqs(src_session_his,
                                             max_len=max_src_session_his_len)

        return src_session_his
