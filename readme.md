## GSERec
This is the official implementation of **GSERec**.

---

## Doanload Data

Download the datasets from the following sources:

- [Qilin](https://hf-mirror.com/datasets/THUIR/Qilin)
- [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

Run the provided Jupyter notebooks in each dataset folder to process the data.

---

## Setup Environment

Install the required dependencies listed in `requirements.txt`.

---


## User Preference Summarization

### Generate Recommendation Preferences

```bash
cd modeling/
CUDA_VISIBLE_DEVICES=0 python generation/LLM_Infer.py --dataset Qilin --reason_task rec
```

### Generate Search Preferences

```bash
cd modeling/
CUDA_VISIBLE_DEVICES=0 python generation/LLM_Infer.py --dataset Qilin --reason_task src
```


The generated preference files will be saved at paths like:
```
data/Qilin/generate/rec_prefer/0-20000.json
data/Qilin/generate/src_prefer/0-20000.json
```


### User Preference Encoding

Encode user preferences using the following command:

```bash
cd modeling/
CUDA_VISIBLE_DEVICES=0 python emb/get_user_emb.py --dataset Qilin --llm_rec_reason generate/rec_prefer/0-20000.json  --llm_src_reason generate/src_prefer/0-20000.json --emb_model_name bge-m3
```

The generated user embeddings will be saved in:
```
data/Qilin/emb/rec_prefer/bge-m3_user_rec.pt
data/Qilin/emb/src_prefer/bge-m3_user_src.pt
```

---

## User Preference Quantization

Quantize user preferences with:
```bash
cd index/
python run_index.py
```

In `run_index.py`, make sure to set:
- `dataset = Qilin`
- `rec_emb = data/Qilin/emb/rec_prefer/bge-m3_user_rec.pt`
- `src_emb = data/Qilin/emb/src_prefer/bge-m3_user_src.pt`

The quantized user codes will be saved in:
```
data/Qilin/index/rec.json
data/Qilin/index/src.json
```

--- 

## GSERec Training

Train the GSERec model using:
```bash
cd modeling/
python run_main.py
```

In `run_main.py`, set:
- `dataset = Qilin`
- `user_rec_index = data/Qilin/index/rec.json`
- `user_src_index = data/Qilin/index/src.json`



### Monitoring Training and Evaluation
Training logs and evaluation results can be found in: `modeling/output/Qilin/GSERec/logs/time.log`.


## Experimental Environment

We conducted the experiments based on the following environments:
* **CUDA Version**: 12.4
* **Operating System**: CentOS Linux release 7.4.1708 (Core)
* **GPU**: NVIDIA® A6000 × 2
* **CPU**: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz