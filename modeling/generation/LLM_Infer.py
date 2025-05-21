import sys

sys.path.append('.')

import argparse
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from generation.dataset import *


def post_process(results):
    processed_results = []
    for index, generate_data in enumerate(results):

        begin_index = generate_data.find(": \"")
        if begin_index == -1:
            begin_index = generate_data.find(": “")
            begin_index += len(": “")
        else:
            begin_index += len(": \"")

        end_index = generate_data[begin_index:].find("\"\n}")
        if end_index == -1:
            end_index = generate_data[begin_index:].find("”\n}")
        if end_index == -1:
            end_index = generate_data[begin_index:].find("\"}")
        if end_index == -1:
            end_index = generate_data[begin_index:].find("\" }")

        end_index += begin_index

        processed_predict = generate_data[begin_index:end_index]

        if len(processed_predict) == 0:
            processed_results.append(generate_data.strip())
        else:
            processed_results.append(processed_predict.strip())

    return processed_results


def post_process_deepseek(results):
    processed_results = []
    for idx, generate_data in enumerate(results):
        try:
            processed_predict = generate_data.split('</think>')[1]
            processed_results.append(processed_predict)
        except:
            print("length overflow: {}".format(idx))
            processed_results.append("")
    return processed_results


def get_text_template(tokenizer: AutoTokenizer, prompt):
    message = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(message,
                                         tokenize=False,
                                         add_generation_prompt=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=2025)

    parser.add_argument("--dataset", type=str, default='Qilin')
    parser.add_argument("--reason_task",
                        type=str,
                        default='rec',
                        choices=['rec', 'src'])

    parser.add_argument("--model_name",
                        type=str,
                        default="DeepSeek-R1-Distill-Qwen-7B",
                        choices=['DeepSeek-R1-Distill-Qwen-7B'])

    parser.add_argument("--max_model_len", type=int, default=10000)

    parser.add_argument("--max_src_session_his_len", type=int, default=20)
    parser.add_argument("--max_session_item_len", type=int, default=1)
    parser.add_argument("--max_rec_his_len", type=int, default=20)

    parser.add_argument("--max_query_len", type=int, default=32)
    parser.add_argument("--max_doc_len", type=int, default=128)

    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=10000)

    return parser


if __name__ == "__main__":
    # cmd: CUDA_VISIBLE_DEVICES=0 python generation/LLM_Infer.py
    start = datetime.now()

    parser = parse_args()
    args = parser.parse_args()

    args.model_path = 'LLMs/{}'.format(args.model_name)
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config = generation_config.to_dict()

    if args.reason_task == 'rec':
        reasonDataset = RecReasonDataset(args, tokenizer)
        max_his_len = args.max_rec_his_len
    elif args.reason_task == 'src':
        reasonDataset = SrcReasonDataset(args, tokenizer)
        max_his_len = args.max_src_session_his_len
    else:
        raise NotImplementedError

    reason_prompts = []
    reason_data_list = []

    for i in tqdm(range(len(reasonDataset))):
        cur_data = reasonDataset[i]
        reason_data_list.append(cur_data)
        reason_prompts.append(get_text_template(tokenizer, cur_data['input']))

    llm = LLM(model=args.model_path,
              gpu_memory_utilization=0.9,
              dtype=torch.float16,
              max_model_len=args.max_model_len)

    sampling_params = SamplingParams(
        seed=args.random_seed,
        temperature=generation_config['temperature']
        if 'temperature' in generation_config.keys() else 0.,
        top_p=generation_config['top_p']
        if 'top_p' in generation_config.keys() else 1.0,
        top_k=generation_config['top_k']
        if 'top_k' in generation_config.keys() else -1,
        repetition_penalty=generation_config['repetition_penalty']
        if 'repetition_penalty' in generation_config.keys() else 1.0,
        best_of=1,
        max_tokens=args.max_new_tokens,
        use_beam_search=False)
    print(sampling_params)

    model_outputs = llm.generate(reason_prompts, sampling_params)
    model_preds = [x.outputs[0].text for x in model_outputs]

    processed_preds = post_process_deepseek(model_preds)
    processed_preds = post_process(processed_preds)

    results = []
    for i in range(len(reason_data_list)):
        cur_result = reason_data_list[i]
        cur_result['prompt'] = reason_prompts[i]
        cur_result['output'] = model_preds[i]
        cur_result['predict'] = processed_preds[i]
        results.append(cur_result)

    base_path = os.path.join('../data/', args.dataset, 'generate')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    begin_idx = reasonDataset.begin_idx
    end_idx = reasonDataset.end_idx

    cur_time = datetime.now().strftime(r"%Y%m%d")

    result_dir_name = f'{args.model_name}_{args.reason_task}-{max_his_len}_{cur_time}'

    result_dir = os.path.join(base_path, result_dir_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file_name = f'{begin_idx}-{end_idx}'

    with open(os.path.join(result_dir, f'{result_file_name}.json'), 'w') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    print(
        f"save file to: {os.path.join(result_dir, f'{result_file_name}.json')}"
    )

    end = datetime.now()
    print("running used time:{}".format(end - start))
