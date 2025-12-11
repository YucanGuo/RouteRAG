# <img src="figs/logo.png" alt="RouteRAG Logo" width="35" style="vertical-align: middle;"/> RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2512.09487-b31b1b.svg)](https://arxiv.org/abs/2512.09487)

This is the official code release of the following paper:

RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

## 📦 1. Installation

Install all dependencies:

```bash
conda create -n routerag python=3.10
conda activate routerag
pip install -r requirements.txt
```

## 🚀 2. Model Training

### ⚙️ Step 1: Start Services

Set up the environment variables and start all required services:

```bash
# Set GPU device for IE model
export CUDA_VISIBLE_DEVICES=4

# Start the OpenIE model service for graph construction
vllm serve /path/to/OpenIE_model \
    --max_model_len 4096 \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

**Parameter Description:**

- `OpenIE_model`: OpenIE model path (Llama-3.1-8B-Instruct/Llama-3.3-70B-Instruct in our experiments)
- `--max_model_len`: Maximum sequence length
- `--gpu-memory-utilization`: GPU memory utilization limit
- `--port`: Service port for the OpenIE model

In a new terminal, start the retrieval service:

```bash
# Set GPU device for retrieval service
export CUDA_VISIBLE_DEVICES=5

# Start retrieval service
python retrieval_api.py \
    --llm_model_name /path/to/OpenIE_model \
    --llm_base_url http://localhost:8000/v1 \
    --embedding_model_name /path/to/embedding_model \
    --dataset hotpotqa_10k \
    --corpus_path train/RL_dataset/corpus.json \
    --port 8001
```

**Parameter Description:**

- `--llm_base_url`: vLLM service API address
- `--embedding_model_name`: Embedding model path (contriever/NV-Embed-v2 in our experiments)
- `--dataset`: Training corpus name
- `--corpus_path`: Training corpus path
- `--port`: Retrieval service port

### 🎯 Step 2: Two-Stage Reinforcement Learning

Navigate to the train directory:

```bash
cd train
```

Execute the two-stage training script:

```bash
bash train_grpo_grag_two_stages.sh
```

**📝 Note:** Please modify the paths in `train_grpo_grag_two_stages.sh` to match your local setup, including `TRAIN_DATA_DIR`, `TEST_DATA_DIR`, `BASE_MODEL`, `SWITCH_STEP`, `SEED`, `retriever.url`, and `trainer.default_local_dir`.

**Two-Stage Training Process:**

- **Stage 1 (Steps 1-<SWITCH_STEP>)**: EM reward only reinforcement learning training
- **Stage 2 (Steps <SWITCH_STEP+1>-40)**: Add efficiency reward to the training

## 🧪 3. Model Testing

Run the test script in the main directory (need to start the OpenIE model service first):

```bash
python test.py --dataset <dataset_name> \
    --llm_base_url http://localhost:8000/v1 \
    --llm_name /path/to/OpenIE_model \
    --reader_name /path/to/trained/model \
    --embedding_name /path/to/embedding_model \
    --deepsearch
```

## 📁 Directory Structure

```
routerag/
├── routerag/                          # Main routerag framework
│   ├── embedding_model/               # Embedding model implementations
│   ├── evaluation/                    # Evaluation implementations
│   ├── information_extraction/        # Information extraction implementations
│   ├── llm/                          # LLM model implementations
│   ├── prompts/                      # Prompt templates and management
│   ├── utils/                        # Utility functions
│   ├── routerag.py                   # Main routerag class
│   ├── embedding_store.py            # Embedding storage management
│   ├── rerank.py                     # Reranking functionality
│   └── README.md                     # routerag module documentation
├── train/                            # Training framework
│   ├── verl/                         # VERL framework
│   ├── routerag/                     # routerag training components
│   │   └── llm_agent/                # LLM agent implementations
│   ├── RL_dataset/                   # Reinforcement learning datasets
│   ├── train_grpo_grag_two_stages.sh # Two-stage training script
│   ├── setup.py                      # Training package setup
│   ├── pyproject.toml                # Python project configuration
│   ├── LICENSE                       # License file
│   └── Notice.txt                    # Notice file
├── dataset/                          # Evaluation datasets
├── outputs/                          # Output directory for results
├── retrieval_api.py                  # FastAPI retrieval service
├── test.py                           # Testing script
├── requirements.txt                  # Main dependencies
└── README.md                         # This file
```

## 🙏 Acknowledgement

We would like to thank the following projects for their foundational work:

- **[Search-R1](https://github.com/PeterGriffinJin/Search-R1)**: We built upon their reinforcement learning framework for model training.
- **[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)**: Our graph retriever is implemented based on HippoRAG 2.

## 📚 Citation

If you find our work useful, please kindly cite:
```
@article{guo2025routerag,
    title={RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning},
    author={Yucan Guo and Miao Su and Saiping Guan and Zihao Sun and Xiaolong Jin and Jiafeng Guo and Xueqi Cheng},
    journal={arXiv preprint arXiv:2512.09487},
    year={2025}
}
```