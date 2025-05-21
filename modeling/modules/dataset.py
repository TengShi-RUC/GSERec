from typing import Any, List

import torch
from modules import const
from modules.sampler import *
from torch.utils.data.dataset import Dataset


class BaseDataSet(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return self.sampler.data.shape[0]

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def collate_batch(self, feed_dicts: List[dict]) -> dict:
        result_dict = dict()
        for key in feed_dicts[0].keys():
            if isinstance(feed_dicts[0][key], List):
                stack_val = list(
                    torch.tensor(list(elem))
                    for elem in zip(*[d[key] for d in feed_dicts]))
                if len(stack_val) == 1:
                    stack_val = stack_val[0]
            elif isinstance(feed_dicts[0][key], torch.Tensor):
                value_tensors = [d[key].unsqueeze(0) for d in feed_dicts]
                stack_val = torch.cat(value_tensors, dim=0)
            elif isinstance(feed_dicts[0][key], str):
                stack_val = [d[key] for d in feed_dicts]
            elif feed_dicts[0][key] is None:
                stack_val = None
            else:
                continue
            result_dict[key] = stack_val
        result_dict['batch_size'] = len(feed_dicts)

        if 'search' in feed_dicts[0].keys():
            result_dict['search'] = feed_dicts[0]['search']

        return result_dict


class RecDataSet(BaseDataSet):

    def __init__(self, train, user_vocab) -> None:
        super().__init__()
        if train == 'train':
            self.sampler = Sampler(data_path=const.rec_train,
                                   user_vocab=user_vocab)
        elif train == 'val':
            self.sampler = Sampler(data_path=const.rec_val,
                                   user_vocab=user_vocab)
        elif train == 'test':
            self.sampler = Sampler(data_path=const.rec_test,
                                   user_vocab=user_vocab)

    def __getitem__(self, index) -> Any:
        return self.sampler.sample(index)
