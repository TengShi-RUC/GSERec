import os
import subprocess
from datetime import datetime

dataset = "Qilin"
CUDA_VISIBLE_DEVICES = "1"

if dataset in ['Amazon_CDs', 'Amazon_Electronics']:
    l2 = 1e-5
elif dataset in ['Qilin']:
    l2 = 1e-7

arguments_dict = {
    # Global
    'random_seed': 2025,
    'time': datetime.now().strftime(r"%Y%m%d-%H%M%S"),
    'train': 1,
    'test_path': '\"\"',
    "data": dataset,
    # Runner
    'epoch': 100,
    'lr': 1e-3,
    'lr_scheduler': 0,
    'min_lr': 1e-6,
    'patience': 3,
    'early_stop': 5,
    'l2': l2,
    'batch_size': 1024,
    'eval_batch_size': 256,
    'optimizer': 'Adam',
    'num_workers': 8,
    'print_interval': 100,
    # Model
    'model_path': '\"\"',
    'dropout': 0.1,
    'num_layers': 1,
    'num_heads': 2,
    'num_gnn_layers': 2,
    'user_rec_index': '',
    'user_src_index': '',
    'user_cl_temp': 0.1,
    'user_cl_weight': 0.1,
    'code_his_cl_temp': 0.1,
    'code_his_cl_weight': 0.01,
}

printDir = "output/{}/{}/logs/".format(arguments_dict['data'], "GSERec")
printFile = os.path.join(printDir, "{}.log".format(arguments_dict['time']))

if not os.path.exists(printDir):
    os.makedirs(printDir, exist_ok=True)

cmd = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} nohup python -u main.py'

for k, v in arguments_dict.items():
    if v is not None:
        cmd += f' --{k} {v}'

cmd += " > {} 2>&1 ".format(printFile)

print("running cmd: ", cmd)
start = datetime.now()
p = subprocess.Popen(cmd, shell=True)
p.wait()

end = datetime.now()
print("runnning used time:{}".format(end - start))
