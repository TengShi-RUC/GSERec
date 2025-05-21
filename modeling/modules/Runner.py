import gc
import logging
import time
from typing import Dict, List

import numpy as np
import torch
from models import BaseModel
from modules import utils
from torch.utils.data import DataLoader

from .dataset import *
from .utils import hit_k, ndcg_k


class BaseRunner(object):

    @staticmethod
    def parse_runner_args(parser):

        parser.add_argument('--epoch',
                            type=int,
                            default=1,
                            help='Number of epochs.')

        parser.add_argument('--lr',
                            type=float,
                            default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--lr_scheduler', type=int, default=0)
        parser.add_argument('--min_lr', type=float, default=1e-6)
        parser.add_argument(
            '--patience',
            type=int,
            default=3,
            help=
            'Number of epochs with no improvement after which learning rate will be reduced'
        )
        parser.add_argument(
            '--early_stop',
            type=int,
            default=5,
            help='The number of epochs when dev results drop continuously.')

        parser.add_argument('--l2',
                            type=float,
                            default=1e-6,
                            help='Weight decay in optimizer.')

        parser.add_argument('--batch_size',
                            type=int,
                            default=1024,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size',
                            type=int,
                            default=512,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer',
                            type=str,
                            default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument(
            '--num_workers',
            type=int,
            default=8,
            help='Number of processors when prepare batches in DataLoader')

        parser.add_argument('--print_interval', type=int, default=100)

        return parser

    def __init__(self, args, model: BaseModel, all_vocabs: Dict) -> None:
        self.args = args

        self.data: str = args.data
        self.model = model

        self.epoch = args.epoch
        self.test_epoch = -1
        self.print_interval = args.print_interval

        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = 1

        self.topk = [1, 5, 10, 20]

        self.metrics = ['NDCG', 'HR']

        self.main_metric = 'NDCG@5'

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None

        self.query_vocab = all_vocabs['query_vocab']
        self.user_vocab = all_vocabs['user_vocab']

    def _build_optimizer(self, model: BaseModel):
        self.optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.customize_parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2)

        if self.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.patience,
                min_lr=self.min_lr)

    def getDataLoader(self, dataset: BaseDataSet, batch_size: int,
                      shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=batch_size // self.num_workers + 1,
            worker_init_fn=utils.worker_init_fn,
            persistent_workers=True,
            collate_fn=dataset.collate_batch)
        return dataloader

    def set_dataloader(self):
        self.train_loader = self.getDataLoader(self.traindata,
                                               batch_size=self.batch_size,
                                               shuffle=True)
        self.val_loader = self.getDataLoader(self.valdata,
                                             batch_size=self.eval_batch_size,
                                             shuffle=False)
        self.test_loader = self.getDataLoader(self.testdata,
                                              batch_size=self.eval_batch_size,
                                              shuffle=False)

    def train(self, model: BaseModel):
        self._build_optimizer(model)

        main_metric_results, dev_results = list(), list()
        last_lr = self.learning_rate

        for epoch in range(self.epoch):
            gc.collect()
            torch.cuda.empty_cache()

            epoch_loss = self.train_epoch(epoch, model)

            logging.info("epoch:{} mean loss:{:.4f}".format(epoch, epoch_loss))

            dev_result, main_result = self.evaluate(model, 'val')
            dev_results.append(dev_result)
            main_metric_results.append(main_result)
            logging.info("Dev Result:")
            logging.info(utils.format_metric(dev_result))

            if self.lr_scheduler:
                self.scheduler.step(main_result)
                new_lr = self.scheduler.get_last_lr()[0]
                if last_lr != new_lr:
                    logging.info("reducing lr from:{} to:{}".format(
                        last_lr, new_lr))
                    last_lr = new_lr

            if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                test_result, _ = self.evaluate(model, 'test')
                logging.info("Test Result:")
                logging.info(test_result)

            if max(main_metric_results) == main_metric_results[-1]:
                model.save_model()
                test_result, _ = self.evaluate(model, 'test')
                logging.info("Test Result:")
                logging.info(utils.format_metric(test_result))

            if self.early_stop > 0 and self.eval_termination(
                    main_metric_results):
                logging.info('Early stop at %d based on dev result.' %
                             (epoch + 1))
                break

        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info("")
        logging.info("Best Dev Result at epoch:{}".format(best_epoch))
        logging.info(utils.format_metric(dev_results[best_epoch]))

        model.load_model()

        test_result, _ = self.evaluate(model, 'test')
        logging.info("")
        logging.info("Test Result:")
        logging.info(utils.format_metric(test_result))

    def train_epoch(self, epoch: int, model: BaseModel):
        model.train()
        logging.info(" ")
        logging.info("Epoch: {}".format(epoch))

        loss_list = []
        loss_dict = {}
        start = time.time()
        for step, batch in enumerate(self.train_loader):
            loss = model.loss(utils.batch_to_gpu(batch, model.device))

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for k, v in loss.items():
                if k in loss_dict.keys():
                    loss_dict[k].append(v.item())
                else:
                    loss_dict[k] = [v.item()]

            loss_list.append(total_loss.item())

            if step > 0 and step % self.print_interval == 0:
                logging.info("epoch:{:d} step:{:d} time:{:.2f}s {}".format(
                    epoch, step,
                    time.time() - start, " ".join([
                        "{}:{:.4f}".format(k,
                                           np.mean(v).item())
                        for k, v in loss_dict.items()
                    ])))

        logging.info("total time: {:.2f}s".format(time.time() - start))

        return np.mean(loss_list).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    @torch.no_grad()
    def predict(self, model: BaseModel, test_loader: DataLoader,
                print_interval):
        model.eval()

        metrics_results = {}

        start = time.time()
        for step, batch in enumerate(test_loader):

            prediction = model.predict(utils.batch_to_gpu(batch, model.device))
            prediction = prediction.cpu().data.numpy().tolist()

            for row in prediction:
                for m in self.metrics:
                    for k in self.topk:
                        key = '{}@{}'.format(m, k)
                        if key not in metrics_results:
                            metrics_results[key] = []
                        if m == 'HR':
                            metrics_results[key].append(hit_k(row, k))
                        elif m == 'NDCG':
                            metrics_results[key].append(ndcg_k(row, k))
                        else:
                            raise ValueError(
                                'Undefined evaluation metric: {}.'.format(m))

            if step > 0 and step % print_interval == 0:
                logging.info("step:{:d} time:{}s".format(
                    step,
                    time.time() - start))

        logging.info("model evaluate time used:{}s".format(time.time() -
                                                           start))

        metrics_results = {k: np.mean(v) for k, v in metrics_results.items()}
        return metrics_results

    def evaluate(self, model: BaseModel, mode: str):
        if mode == 'val':
            results = self.predict(model,
                                   self.val_loader,
                                   print_interval=self.print_interval)
        elif mode == 'test':
            results = self.predict(model,
                                   self.test_loader,
                                   print_interval=self.print_interval)
        else:
            raise ValueError('test set error')

        return results, results[self.main_metric]

    def build_dataset(self):
        raise NotImplementedError


class RecRunner(BaseRunner):

    def __init__(self, args, model: BaseModel, all_vocabs: Dict) -> None:
        super().__init__(args, model, all_vocabs)
        self.build_dataset()
        self.set_dataloader()

    def build_dataset(self):
        self.traindata = RecDataSet(train='train', user_vocab=self.user_vocab)
        self.valdata = RecDataSet(train='val', user_vocab=self.user_vocab)
        self.testdata = RecDataSet(train='test', user_vocab=self.user_vocab)

        logging.info("rec_train: {} ".format(len(self.traindata)))
        logging.info("rec_val: {}".format(len(self.valdata)))
        logging.info("rec_test: {}".format(len(self.testdata)))
