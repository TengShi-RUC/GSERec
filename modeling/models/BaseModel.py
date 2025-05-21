import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import const, utils

from .Inputs import *


class BaseModel(nn.Module):

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path',
                            type=str,
                            default='',
                            help='Model save path.')
        parser.add_argument('--dropout', type=float, default=0.1)

        return parser

    def __init__(self, args, all_vocabs):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.dropout = args.dropout
        self.batch_size = args.batch_size

        self.args = args

        self.init_emb(args, all_vocabs)
        logging.info("final emb size:{}".format(const.final_emb_size))
        self.user_size = const.final_emb_size
        self.item_size = const.final_emb_size
        self.query_size = const.final_emb_size

        self.loss_fn = self.bce_loss

    def _init_weights(self):
        # weight initialization xavier_normal (a.k.a glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, "{}.pt".format('best'))
        utils.check_dir(model_path)
        logging.info("save model to: {}".format(model_path))
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
            model_path = os.path.join(model_path, "{}.pt".format('best'))
        logging.info("load model from: {}".format(model_path))
        self.load_state_dict(
            torch.load(model_path, map_location=self.device,
                       weights_only=True))

    def count_variables(self) -> int:
        total_parameters = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                num_p = p.numel()
                total_parameters += num_p

        return total_parameters

    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad,
                              self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{
            'params': weight_p
        }, {
            'params': bias_p,
            'weight_decay': 0
        }]
        return optimize_dict

    def bce_loss(self, logits):
        # logits: [batch_size, num_items]
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        logits = F.sigmoid(logits)

        loss = F.binary_cross_entropy(logits, labels)
        return loss

    def loss(self, inputs):
        return self.rec_loss(inputs)

    def predict(self, inputs):
        return self.rec_predict(inputs)

    def rec_loss(self, inputs):
        raise NotImplementedError

    def rec_predict(self, inputs):
        raise NotImplementedError
