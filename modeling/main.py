import argparse
import datetime
import logging
import os

import torch
from models import *
from modules import *
from modules import const, utils


def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default='none')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--data',
                        type=str,
                        default='Qilin',
                        help='Choose a dataset.')
    return parser


if __name__ == '__main__':
    # cmd: python main.py --model GSERec --runner RecRunner --data Qilin
    global_start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = GSERec.parse_model_args(parser)
    parser = RecRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()
    utils.setup_seed(args.random_seed)

    args.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.data == 'Amazon_CDs':
        const.init_setting_Amazon_CDs()
    elif args.data == 'Amazon_Electronics':
        const.init_setting_Amazon_Electronics()
    elif args.data == 'Qilin':
        const.init_setting_Qilin()
    else:
        raise ValueError('Dataset Error')

    if args.time == 'none':
        cur_time = datetime.datetime.now()
        args.time = cur_time.strftime(r"%Y%m%d-%H%M%S")

    output_dir = "output/{}/{}/".format(args.data, 'GSERec')

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    utils.printSetting()

    args.model_path = os.path.join(output_dir,
                                   "checkpoints/{}".format(args.time))

    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {}'.format(flag, value))

    all_vocabs = utils.load_vocabs()
    model: BaseModel = GSERec(args, all_vocabs)
    runner: BaseRunner = RecRunner(args, model, all_vocabs)

    print(model)
    num_parameters = model.count_variables()
    logging.info("num model parameters:{}".format(num_parameters))

    if args.train == 0:
        if args.test_path != '':
            model.load_model(model_path=args.test_path)

        test_result, _ = runner.evaluate(model, 'test', print_group=False)
        logging.info("")
        logging.info("Test Result:")
        logging.info(utils.format_metric(test_result))

    else:
        runner.train(model)

    global_end_time = datetime.datetime.now()
    print("runnning used time:{}".format(global_end_time - global_start_time))
