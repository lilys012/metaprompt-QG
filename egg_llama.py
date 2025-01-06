from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import LlamaGenModel, LlamaProtoGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models as stmodels
from sentence_transformers.readers import InputExample
from transformers import LlamaTokenizer

import pathlib, os
import logging
import math

from time import time
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import random
import sys
import gc
import torch
import numpy as np
from tqdm import tqdm
import json
import csv
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ret_model_path', required=False, default="sentence-transformers/msmarco-distilbert-base-tas-b")
    parser.add_argument('--gen_chat_model_path', required=False, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--gen_base_model_path', required=False, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--method', required=False, default="egg")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--tailor', required=False, default = False, type=bool)
    parser.add_argument('--query_tailor', required=False, default = False, type=bool)
    parser.add_argument('--qgen_prefix', required=False, default = 'gen')
    parser.add_argument('--qpp', required=False, default = 8, type=int)
    parser.add_argument('--doc_size', required=False, default = 100000, type=int)
    parser.add_argument('--batch_size', required=False, default = 75, type=int) 
    args = parser.parse_args()

    qgen_prefix = args.qgen_prefix

    ret_model_path = args.ret_model_path
    ret_model_name = ret_model_path.split('/')[-1] if len(ret_model_path.split('/')) else ret_model_path
    gen_model_path = args.gen_base_model_path
    gen_model_name = gen_model_path.split('/')[-1] if len(gen_model_path.split('/')) else gen_model_path

    dataset = args.dataset
    data_path = args.data_path
    path_to_generated_data = f"experiments/data/{gen_model_path}/{dataset}"
    os.makedirs(path_to_generated_data, exist_ok=True)
    if "corpus.jsonl" not in os.listdir(path_to_generated_data):
        corpus_path = os.path.join(data_path, "corpus.jsonl")
        os.system(f"cp {corpus_path} {path_to_generated_data}")
    assert "corpus.jsonl" in os.listdir(path_to_generated_data)

    #### for few-shot
    split = "train" if dataset in ["fever", "scifact"] else "dev" if dataset in ["dbpedia-entity"] else "test"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    initial_ice_num = 4
    initial_ids = random.sample(list(qrels.keys()), k=initial_ice_num)
    tokenizer = LlamaTokenizer.from_pretrained(gen_model_path, padding_side="left")
    tokenizer.pad_token='[PAD]'
    prefix = ""
    example_ids = []
    for i in range(initial_ice_num):
        while True:
            qid = random.sample(list(qrels.keys()), k=1)[0]
            cid = list(qrels[qid].keys())[0]
            if cid in corpus: break
        encodings = tokenizer(corpus[cid]["title"] + " " + corpus[cid]["text"], truncation=True, return_tensors="pt", max_length=350)
        decoded_text = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
        prefix += "Passage: " + decoded_text + f"\n{prompt[dataset]}: " + queries[qid] + "\n\n"
        example_ids.append((qid, cid))
    with open(f"initial_ids/{dataset}.json", "w") as f: json.dump(example_ids, f, indent=4)

    model_save_path = os.path.join(f"experiments/model/{gen_model_path}/", "{}-{}-{}5".format(ret_model_name, dataset, qgen_prefix))
    os.makedirs(model_save_path, exist_ok=True)
    
    if 'pytorch_model.bin' not in os.listdir(model_save_path):
        os.makedirs(path_to_generated_data, exist_ok=True)
        ques_per_passage = int(args.qpp)
        corpus_ids = None
        proto_qgen_prefix = qgen_prefix+("-proto" if args.method == "egg" else "-few" if args.method == "few" else "-base")

        #### generated queries exist
        if f"{proto_qgen_prefix}-qrels" in os.listdir(
                path_to_generated_data
            ) and f"{proto_qgen_prefix}-queries.jsonl" in os.listdir(path_to_generated_data):
            logger.info("Loading from existing generated data")
        else:
            #### generation
            if args.method == "few":
                print("Generate few-shot queries")
                model = model = LlamaGenModel(gen_model_path, gen_prefix=prefix, dataset=dataset)
            elif args.method == "base":
                print("Generate baseline queries")
                model = LlamaGenModel(gen_model_path, gen_prefix=f"Read the passage and generate a query.", dataset=dataset)
            else:
                print("Generate proto queries")
                model = LlamaProtoGenModel(args.gen_chat_model_path, dataset=dataset)
        
            pool = model.start_multi_process_pool()
            generator = QGen(model=model)
            corpus_ids = generator.generate_multi_process(corpus, pool, output_dir=path_to_generated_data, ques_per_passage=1, prefix=proto_qgen_prefix, batch_size=10, doc_size=int(args.doc_size))

            with torch.no_grad():
                model = None
                generator = None
                gc.collect()
                torch.cuda.empty_cache()

        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=proto_qgen_prefix).load(split="train")            
        print("Corpus: " + str(len(corpus)) + " Queries: " + str(len(gen_queries)) + " Qrels: " + str(len(gen_qrels)))

        if args.tailor or args.query_tailor:
            if f"{qgen_prefix}-qrels" in os.listdir(
                path_to_generated_data
            ) and f"{qgen_prefix}-queries.jsonl" in os.listdir(path_to_generated_data):
                logger.info("Loading from existing tailored data")
            else:
                print("Generate tailored queries")
                cid_to_query = {}
                for qid, item in gen_qrels.items():
                    for cid, _ in item.items():
                        if cid not in cid_to_query: cid_to_query[cid] = gen_queries[qid]

                model = LlamaGenModel(gen_model_path, gen_prefix="", dataset=dataset)
                pool = model.start_multi_process_pool()
                generator = QGen(model=model)
                generator.generate_multi_process(corpus, pool, output_dir=path_to_generated_data, ques_per_passage=ques_per_passage, prefix=qgen_prefix, batch_size=1, doc_size=int(args.doc_size), cid_to_query=cid_to_query, corpus_ids=corpus_ids)
                print(generator.insidecnt)

            corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix).load(split="train")

        evaluation_steps = 5000
        warmup_steps = 1000
        batch_size = int(args.batch_size)

        #### Provide any HuggingFace model and fine-tune from scratch
        model = SentenceTransformer(ret_model_path)
        retriever = TrainRetriever(model=model, batch_size=batch_size) 
        train_samples, id2qid = retriever.load_train(corpus, gen_queries, gen_qrels)

        #### Please Note - not all datasets contain a dev split, comment out the line if such the case
        if f"qrels/dev.tsv" in os.listdir(path_to_generated_data):
            dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
            ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
        else:
            ir_evaluator = retriever.load_dummy_evaluator()

        #### Prepare training samples
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
        num_epochs = 3 if len(corpus) <= 60000 else 1
        retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                        evaluator=ir_evaluator, 
                        epochs=num_epochs,
                        output_path=model_save_path,
                        warmup_steps=warmup_steps,
                        evaluation_steps=evaluation_steps,
                        use_amp=True)
    else:
        model = SentenceTransformer(model_save_path)

    #### Evaluation
    test_corpus, test_queries, test_qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    test_corpus = corpus

    sbert = models.SentenceBERT(sep=" ")
    sbert.q_model = model
    sbert.doc_model = model
    test_model = DRES(sbert, batch_size=256, corpus_chunk_size=10000, show_progress_bar=False)
    test_retriever = EvaluateRetrieval(test_model, score_function="dot")

    print("Evaluating... Used examples from "+split)
    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = test_retriever.retrieve(test_corpus, test_queries, split=split, method=args.method)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(test_retriever.k_values))
    ndcg, _map, recall, precision = test_retriever.evaluate(test_qrels, results, test_retriever.k_values, ignore_identical_ids=False)

    print(ndcg["NDCG@10"])

    with open(path_to_generated_data+"/scores_" + qgen_prefix + ".json", "w") as f10:
        json.dump([ndcg, _map, recall, precision], f10, indent=4)
