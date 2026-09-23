# EDITDAVR: MULTI-HOP KNOWLEDGE EDITING VIA DIALECTICAL AGENTIC VERIFICATION AND REFLECTION

## Project Run Guide

### 1. Environment Setup

```bash
conda create -n davr python=3.9
conda activate davr
pip install -r requirements.txt
python -m spacy_entity_linker "download_knowledge_base"
```

### 2. Notes

（1）**Model Deployment**: Our method uses an API format to deploy the model. Refer to the documentation at [FastChat](https://github.com/lm-sys/FastChat).

（2）**Pre-trained Models**: You need to download the `distilbert-base-cased` and `rebel-large` models from HuggingFace and place them in the project directory "./model":


- [distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased)

- [rebel-large](https://huggingface.co/Babelscape/rebel-large)


（3）**Datasets and vLLM Models**: 

- **Answer + Vertification**:

python -m vllm.entrypoints.openai.api_server   --model ../Vicuna-7B   --served-model-name vicuna-7b   --port 7002   --max-model-len 2048   --gpu-memory-utilization 0.85   --dtype auto --chat-template vicuna_template.jinja

```python
ANSWER_API_BASE = "https://localhost:7001/v1"   
ANSWER_API_KEY = ""
ANSWER_MODEL = "llama-2-7b"


VERIFY_API_BASE = "http://localhost:7002/v1"
VERIFY_API_KEY = ""
VERIFY_MODEL = "qwen3-8b" 

```

- **Divide Model**:

```python
DIVIDE_API_BASE = ""    
DIVIDE_API_KEY = ""
DIVIDE_MODEL = "deepseek-v4-flash"   
```


vLLM needs to be installed separately depending on the CUDA version and system environment.

（4）**Run the Experiment**:
Use the following command to execute the experiment:

```
cd ./test
python dynamic_exp_bridge7.py --dataset MQuAKE-CF-3k --edit 1
python dynamic_exp_bridge7.py --dataset MQuAKE-CF-3k --edit 100  
python dynamic_exp_bridge7.py --dataset MQuAKE-CF-3k --edit 3000
```

Here, `N` represents the batch size for editing.



