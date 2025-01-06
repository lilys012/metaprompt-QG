from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import FlanGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models as stmodels
from sentence_transformers.readers import InputExample

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
    parser.add_argument('--gen_model_path', required=False, default="google/flan-t5-xl")
    parser.add_argument('--filter_model_path', required=False, default="sentence-transformers/msmarco-distilbert-base-tas-b")
    parser.add_argument('--do_filter', required=False, default = False, type=bool) 
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--qgen_prefix', required=False, default = 'gen')
    parser.add_argument('--qpp', required=False, default = 8, type=int)
    parser.add_argument('--doc_size', required=False, default = 100000, type=int)
    parser.add_argument('--cos_thresh', required=False, default = 0.25, type=float) # How much thresh to cut
    parser.add_argument('--cos_perc', required=False, default = 1.0, type=float) # How much % to leave
    parser.add_argument('--qrels', required=False, default = "") 
    parser.add_argument('--batch_size', required=False, default = 75, type=int) 
    args = parser.parse_args()

    ret_model_path = args.ret_model_path
    ret_model_name = ret_model_path.split('/')[-1] if len(ret_model_path.split('/')) else ret_model_path
    gen_model_path = args.gen_model_path
    gen_model_name = gen_model_path.split('/')[-1] if len(gen_model_path.split('/')) else gen_model_path
    filter_model_path = args.filter_model_path
    filter_model_name = filter_model_path.split('/')[-1] if len(filter_model_path.split('/')) else filter_model_path

    do_filter = args.do_filter
    consistent = args.consistent
    qgen_prefix = args.qgen_prefix

    dataset = args.dataset
    data_path = f"dataset/{dataset}"
    path_to_generated_data = f"experiments/{gen_model_name}/{dataset}"
    os.makedirs(path_to_generated_data, exist_ok=True)
    if "corpus.jsonl" not in os.listdir(path_to_generated_data):
        corpus_path = os.path.join(data_path, "corpus.jsonl")
        os.system(f"cp {corpus_path} {path_to_generated_data}")

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f"experiments/models/{gen_model_path}/", "{}-{}-{}".format(ret_model_name, dataset, qgen_prefix))
    os.makedirs(model_save_path, exist_ok=True)

    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    assert "corpus.jsonl" in os.listdir(path_to_generated_data)
    corpus = GenericDataLoader(path_to_generated_data).load_corpus()

    if 'pytorch_model.bin' not in os.listdir(model_save_path):

        ##############################
        #### 1. Query-Generation  ####
        ##############################

        os.makedirs(path_to_generated_data, exist_ok=True)
        ques_per_passage = int(args.qpp)

        prompt = {"arguana":"Argument", "scifact":"Claim", "fever":"Claim", "scidocs":"Title", "dbpedia-entity":"Entity"}

        #### generated queries exist
        if f"{qgen_prefix}-qrels" in os.listdir(
                path_to_generated_data
            ) and f"{qgen_prefix}-queries.jsonl" in os.listdir(path_to_generated_data):
            logger.info("Loading from existing generated data")
        else:
            #### generation
            model = FlanGenModel(gen_model_path, gen_prefix=f"Write a {prompt[dataset]} related to topic of the passage. Do not directly use wordings from the passage. passage: ", dataset=dataset)
            pool = model.start_multi_process_pool()
            generator = QGen(model=model)
            generator.generate_multi_process(corpus, pool, output_dir=path_to_generated_data, ques_per_passage=ques_per_passage, prefix=qgen_prefix, batch_size=4, doc_size=int(args.doc_size))

            with torch.no_grad():
                model = None
                generator = None
                gc.collect()
                torch.cuda.empty_cache()

        if args.qrels != "": #### Load previously generated qrel file
            corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix, qrels_file=args.qrels).load(split="train")  
        else:
            corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix).load(split="train")            

        ################################
        #### 2. Train Dense-Encoder ####
        ################################

        #### Configure Train params
        num_epochs = 1
        evaluation_steps = 5000
        warmup_steps = 1000
        batch_size = int(args.batch_size)

        filter_word_embedding_model = stmodels.Transformer(filter_model_path, max_seq_length=350)
        filter_pooling_model = stmodels.Pooling(filter_word_embedding_model.get_word_embedding_dimension())
        filter_model = SentenceTransformer(modules=[filter_word_embedding_model, filter_pooling_model])
        filter_retriever = TrainRetriever(model=filter_model, batch_size=batch_size)
        train_samples, id2qid = filter_retriever.load_train(corpus, gen_queries, gen_qrels)

        if do_filter:
            if consistent or qgen_prefix+"-cosine.tsv" not in os.listdir(path_to_generated_data):
                train_dataloader = filter_retriever.prepare_train(train_samples, shuffle=True)
                train_loss = losses.MultipleNegativesRankingLoss(model=filter_retriever.model)
                if f"qrels/dev.tsv" in os.listdir(path_to_generated_data):
                    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
                    ir_evaluator = filter_retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
                else:
                    ir_evaluator = filter_retriever.load_dummy_evaluator()
                filter_retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                                evaluator=ir_evaluator, 
                                epochs=num_epochs,
                                output_path=None,
                                warmup_steps=warmup_steps,
                                evaluation_steps=evaluation_steps,
                                use_amp=True)
            filtered_train_samples = []
            filter_qrels = {}

            samples_cosine = []
            print(f"Filtering threshold={float(args.cos_thresh)}, {float(args.cos_perc)*100}%")

            if qgen_prefix+"-cosine.tsv" not in os.listdir(path_to_generated_data):
                for i in tqdm(range(0, len(train_samples), batch_size)):
                    batch_train_samples = train_samples[i:min(i+batch_size, len(train_samples))]
                    queries = [batch_train_samples[j].texts[0] for j in range(len(batch_train_samples))]
                    documents = [batch_train_samples[j].texts[1] for j in range(len(batch_train_samples))]
                    with torch.no_grad():
                        q_encodings = filter_model.encode(queries, show_progress_bar=False) # (batch_size * 768)
                        d_encodings = filter_model.encode(documents, show_progress_bar=False) # (batch_size * 768)
                        q_norm = np.linalg.norm(q_encodings, axis=1)
                        d_norm = np.linalg.norm(d_encodings, axis=1)
                        dot_product = np.sum(q_encodings * d_encodings, axis=1)
                        cosine = dot_product / q_norm / d_norm
                    samples_cosine.extend(zip(cosine, batch_train_samples))

                def sortkey(samples): return samples[0]
                samples_cosine.sort(key=sortkey)

                # save entire cosine scores
                f = open(path_to_generated_data+"/"+qgen_prefix+"-cosine.tsv", "w")
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                for c, sample in samples_cosine:
                    writer.writerow([c, sample.guid, sample.label, sample.texts[0], sample.texts[1]])
                f.close()
            else:
                reader = csv.reader(open(path_to_generated_data+"/"+qgen_prefix+"-cosine.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                    
                for row in reader:
                    c, guid, lbl, q, t = float(row[0]), row[1], int(row[2]), row[3], row[4]
                    samples_cosine.append([c, InputExample(guid=guid, texts=[q, t], label=lbl)])

            flag = 0
            low_threshold = int(len(samples_cosine)*float(args.cos_perc))
            for c, sample in samples_cosine:
                if c > float(args.cos_thresh):
                    flag = low_threshold + 1
                else:
                    flag += 1
                if flag > low_threshold:
                    filter_qrels[sample.guid] = {id2qid[sample.guid]:1}
                    filtered_train_samples.append(sample)
            filter_qrels_file = os.path.join(path_to_generated_data, qgen_prefix + "-qrels", f"train_filter_{str(int(args.cos_perc*100))}_{str(int(args.cos_thresh*100))}.tsv")
            util.write_to_tsv(output_file=filter_qrels_file, data=filter_qrels)
            print(f"Before: {len(train_samples)}, After: {len(filtered_train_samples)}")
            del train_samples, id2qid, samples_cosine, filter_qrels
            filter_model.cpu()
            del filter_model, filter_retriever
    else:
        filtered_train_samples = train_samples

        #### Provide any HuggingFace model and fine-tune from scratch
        word_embedding_model = stmodels.Transformer(ret_model_path, max_seq_length=350)
        pooling_model = stmodels.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=torch.device('cuda'))
        model = SentenceTransformer(ret_model_path)

        #### Provide any sentence-transformers model path
        retriever = TrainRetriever(model=model, batch_size=batch_size) 

        #### Please Note - not all datasets contain a dev split, comment out the line if such the case
        if f"qrels/dev.tsv" in os.listdir(path_to_generated_data):
            dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
            ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
        else:
            ir_evaluator = retriever.load_dummy_evaluator()

        #### Prepare training samples
        train_dataloader = retriever.prepare_train(filtered_train_samples, shuffle=True)
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
        n = len(filtered_train_samples)
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

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = test_retriever.retrieve(test_corpus, test_queries, method="egg")
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(test_retriever.k_values))
    ndcg, _map, recall, precision = test_retriever.evaluate(test_qrels, results, test_retriever.k_values, ignore_identical_ids=False)
    
    with open(path_to_generated_data+"/scores_" + qgen_prefix + ".json", "w") as f10:
        json.dump([ndcg, _map, recall, precision], f10, indent=4)
