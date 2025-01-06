# Meta-prompting Task-Adaptive Query Generator

This is the repository for the paper [Disentangling Questions from Query Generation for Task-Adaptive Retrieval](https://arxiv.org/pdf/2409.16570) (EMNLP 2024 Findings).

### Setup

Please install necessary packages through the command below.

```bash
pip install -r requirements.txt
```

## Dataset

EGG accepts data in the [BeIR](https://github.com/beir-cellar/beir) format. Please make sure the dataset is in `dataset` directory.

## Algorithm

Here we describe the summary of our method. First, documents are fed into instruction-following LM (FLAN-T5-XL / Llama2) with their search intent in the prompt. After queries are generated, together with the documents they form a synthetic dataset. EGG-FLAN applies filtering mechanism based on cosine similarity or removing low-quality pairs. EGG-LLAMA performs in-context learning with prototype pairs.

## Usage

### EGG + DPR

#### EGG-FLAN

[egg_flan.py](egg_flan.py) contains code to generate queries with FLAN-T5 model and train retriever.

Entire pipeline can be easily run with the following command.

```bash
python egg_flan.py \
    --ret_model_path sentence-transformers/msmarco-distilbert-base-tas-b \ # retriever
    --gen_model_path google/flan-t5-xl \ # generator
    --filter_model_path sentence-transformers/msmarco-distilbert-base-tas-b \ # filtering retriever
    --do_filter True \ # conduct filtering or not
    --dataset scifact \ # dataset
    --qgen_prefix gen-flan \ # prefix
    --qpp 8 \ # queries per passage
    --doc_size 100000 \ # maximum doc size to generate
    --cos_thresh 0.25 \ # filtering threshold
    --batch_size 75 \ # batch size
```

#### EGG-LLAMA

[egg_llama.py](egg_llama.py) contains code to generate queries with Llama2 models and train retriever.

Please make sure you have both Llama2-chat (for prototype queries) and Llama2-base (for in-context learning) models.

Entire pipeline can be easily run with the following command.

```bash
python egg_llama.py \
    --ret_model_path sentence-transformers/msmarco-distilbert-base-tas-b \ # retriever
    --gen_chat_model_path meta-llama/Llama-2-7b-chat-hf \ # proto query generator
    --gen_base_model_path meta-llama/Llama-2-7b-hf \ # proto query generator
    --method egg \ # method (base, egg, few)
    --dataset scifact \ # dataset
    --data_path dataset/scifact \ # data path
    --tailor True \ # indicates to perform in-context learning
    --qgen_prefix gen-llama \ # prefix
    --qpp 8 \ # queries per passage
    --doc_size 100000 \ # maximum doc size to generate
    --batch_size 75
```

### EGG + GPL

We require query generation process above to be performed first.

Please feel free to modify qpp for generating queries for GPL, although we have provided the commented automatic calculation.

To reproduce our experiments, please run the command below.

```bash
python gpl/gpl_train.py \
    --dataset scifact \
    --qgen_prefix gen-flan \
    --path_to_generated_data experiments/scifact
```
