## GSERec
This is the official implementation of GSERec.

## User Preference Summarization

```bash
cd modeling/
CUDA_VISIBLE_DEVICES=0 python generation/LLM_Infer.py
```


## User Preference Quantization

```bash
cd index/
python run_index.py
```


## GSERec Training

```bash
cd modeling/
python run_main.py
```