import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data as data


class EmbCollator:

    def __call__(self, batch):
        src_emb = torch.cat([x['src'].unsqueeze(0) for x in batch], dim=0)
        batch_size = len(src_emb)

        rec_emb = torch.cat([x['rec'].unsqueeze(0) for x in batch], dim=0)

        return {"src": src_emb, "rec": rec_emb, "batch_size": batch_size}


class EmbDataset(data.Dataset):

    def __init__(self, args):
        self.src_path = os.path.join("../data", args.dataset, "emb",
                                     args.src_emb)
        self.src_emb = torch.load(self.src_path, weights_only=True).float()
        self.src_dim = self.src_emb.shape[-1]
        logging.info("load src emb: {}".format(self.src_path))

        self.rec_path = os.path.join("../data", args.dataset, "emb",
                                     args.rec_emb)
        self.rec_emb = torch.load(self.rec_path, weights_only=True).float()

        self.rec_dim = self.rec_emb.shape[-1]
        logging.info("load rec emb: {}".format(self.rec_path))

        assert len(self.src_emb) == len(self.rec_emb)

    def __getitem__(self, index):
        src_emb = self.src_emb[index]
        rec_emb = self.rec_emb[index]

        return {"src": src_emb, "rec": rec_emb}

    def __len__(self):
        return len(self.src_emb)
