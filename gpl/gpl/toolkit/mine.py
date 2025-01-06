import json
from gpl.toolkit.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import torch
from easy_elasticsearch import ElasticSearchBM25
import tqdm
import numpy as np
import os
import logging
import argparse
import time
from collections import namedtuple

topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])

logger = logging.getLogger(__name__)


class NegativeMiner(object):
    def __init__(
        self,
        generated_path,
        prefix,
        retrievers=["bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["none", "cos_sim", "cos_sim"],
        nneg=50,
        use_train_qrels: bool = False,
        qrels_file: str = None,
    ):
        if use_train_qrels:
            logger.info("Using labeled qrels to construct the hard-negative data")
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path
            ).load(split="train")
        else:
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path, prefix=prefix, qrels_file=qrels_file
            ).load(split="train")
        self.output_path = os.path.join(generated_path, prefix+"-hard-negatives.jsonl")
        self.retrievers = retrievers
        self.retriever_score_functions = retriever_score_functions
        if "bm25" in retrievers:
            assert (
                nneg <= 10000
            ), "Only `negatives_per_query` <= 10000 is acceptable by Elasticsearch-BM25"
            assert retriever_score_functions[retrievers.index("bm25")] == "none"

        assert set(retriever_score_functions).issubset({"none", "dot", "cos_sim"})

        self.nneg = nneg
        if nneg > len(self.corpus):
            logger.warning(
                "`negatives_per_query` > corpus size. Please use a smaller `negatives_per_query`"
            )
            self.nneg = len(self.corpus)

    def _get_doc(self, did):
        return " ".join([self.corpus[did]["title"], self.corpus[did]["text"]])

    def _mine_sbert(self, model_name, score_function):
        logger.info(f"Mining with {model_name}")
        assert score_function in ["dot", "cos_sim"]
        normalize_embeddings = False
        if score_function == "cos_sim":
            normalize_embeddings = True

        result = {}
        sbert = SentenceTransformer(model_name)
        # pool = sbert.start_multi_process_pool()
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        batch_size=128
        doc_embs = sbert.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=normalize_embeddings,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), batch_size):
            qid_batch = qids[start : start + batch_size]
            qemb_batch = sbert.encode(
                queries[start : start + batch_size], 
                show_progress_bar=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=normalize_embeddings,
            )
            score_mtrx = np.matmul(qemb_batch, np.transpose(doc_embs))  # (qsize, dsize)
            indices_topk = np.argpartition(score_mtrx, -self.nneg, axis=-1)[:, -self.nneg:]
            # score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
            # _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result

    def _mine_bm25(self):
        logger.info(f"Mining with bm25")
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=self.nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())
            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)
            result[qid] = neg_dids
        return result

    def run(self):
        hard_negatives = {}
        for retriever, score_function in zip(
            self.retrievers, self.retriever_score_functions
        ):
            if retriever == "bm25":
                hard_negatives[retriever] = self._mine_bm25()
            else:
                hard_negatives[retriever] = self._mine_sbert(retriever, score_function)

        logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                "qid": qid,
                "pos": list(pos_dids.keys()),
                "neg": {k: v[qid] for k, v in hard_negatives.items()},
            }
            result_jsonl.append(line)

        logger.info(f"Saving data to {self.output_path}")
        with open(self.output_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path")
    args = parser.parse_args()

    miner = NegativeMiner(args.generated_path, "qgen")
    miner.run()
