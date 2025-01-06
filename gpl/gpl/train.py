import shutil
# from beir.datasets.data_loader import GenericDataLoaderx
from .toolkit import (
    qgen,
    NegativeMiner,
    MarginDistillationLoss,
    GenerativePseudoLabelingDataset,
    PseudoLabeler,
    evaluate,
    resize,
    load_sbert,
    set_logger_format,
    mnrl,
    save_queries,
    save_qrels,
    extract_queries_split,
    rescale_gpl_training_data,
    GenericDataLoader,
)
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import os
import logging
import argparse
from typing import List, Union
import math

# import crash_ipdb


set_logger_format()
logger = logging.getLogger(
    "gpl.train"
)  # Here we do not use __name__ to have unified logger name, no matter whether we are using `python -m` or `import gpl; gpl.train`

dataset = "msmarco"
def train(
    path_to_generated_data: str = f"experiments/dataset/flan-t5-xl/{dataset}",
    output_dir: str = f"experiments/gpl/msmarco-distilbert-base-tas-b/{dataset}",
    mnrl_output_dir: str = None,
    mnrl_evaluation_output: str = None,
    do_evaluation: str = True,
    evaluation_data: str = f"dataset/{dataset}",
    evaluation_output: str = f"experiments/gpl/eval/{dataset}",
    qgen_prefix: str = "gen",
    base_ckpt: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
    generator: str = "BeIR/query-gen-msmarco-t5-base-v1",
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size_gpl: int = 32,
    batch_size_generation: int = 32,
    pooling: str = None,
    max_seq_length: int = 350,
    new_size: int = None,
    queries_per_passage: int = 3,
    gpl_steps: int = 140000,
    use_amp: bool = False,
    retrievers: List[str] = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
    retriever_score_functions: List[str] = ["cos_sim", "cos_sim"],
    negatives_per_query: int = 50,
    eval_split: str = "test",
    use_train_qrels: bool = False,
    gpl_score_function: str = "dot",
    rescale_range: List[float] = None,
    qrels_file: str = "train",
):
    #### Assertions ####
    assert pooling in [None, "mean", "cls", "max"]
    if do_evaluation:
        assert evaluation_data is not None
        assert evaluation_output is not None
        try:
            GenericDataLoader(evaluation_data)
        except Exception as e:
            logger.error("Cannot load evaluation data for evaluation usage.")
            raise e
    if new_size is not None and new_size != -1:
        assert new_size * queries_per_passage >= batch_size_gpl

    #### Make sure there is a `corpus.jsonl` file. It should be under either `path_to_generated_data` or `evaluation_data`` ####
    #### Also resize the corpus for efficient training if required  ####
    os.makedirs(path_to_generated_data, exist_ok=True)
    if "corpus.jsonl" not in os.listdir(path_to_generated_data):
        logger.info(
            f"Corpus does not exist in {path_to_generated_data}. Now clone the one from the evaluation path {evaluation_data}"
        )
        assert "corpus.jsonl" in os.listdir(
            evaluation_data
        ), f"No corpus found in evaluation path {evaluation_data}! It should be in the BeIR format. For more details, please refer to https://github.com/UKPLab/beir#beers-available-datasets."
        corpus_path = os.path.join(evaluation_data, "corpus.jsonl")
        os.system(f"cp {corpus_path} {path_to_generated_data}")

    #### Load Synthetic query ####
    assert f"{qgen_prefix}-qrels" in os.listdir(path_to_generated_data)
    assert f"{qgen_prefix}-queries.jsonl" in os.listdir(path_to_generated_data)
    logger.info("Loading from existing generated data")
    corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix, qrels_file=qrels_file).load(split="train")

    #### Hard-negative mining ####
    #### This will be skipped if there is an existing `hard-negatives.jsonl` file under `path_to_generated_data` ####
    if qgen_prefix+"-hard-negatives.jsonl" in os.listdir(path_to_generated_data):
        logger.info("Using exisiting hard-negative data")
    else:
        logger.info("No hard-negative data found. Now mining it")
        miner = NegativeMiner(
            path_to_generated_data,
            qgen_prefix,
            retrievers=retrievers,
            retriever_score_functions=retriever_score_functions,
            nneg=negatives_per_query,
            use_train_qrels=use_train_qrels,
            qrels_file=qrels_file,
        )
        miner.run()

    #### Pseudo labeling ####
    #### This will be skipped if there is an existing `gpl-training-data.tsv` file under `path_to_generated_data` ####
    gpl_training_data_fname = qgen_prefix+"-gpl-training-data.tsv"
    if gpl_training_data_fname in os.listdir(path_to_generated_data):
        logger.info("Using existing GPL-training data")
    else:
        logger.info("No GPL-training data found. Now generating it via pseudo labeling")
        pseudo_labeler = PseudoLabeler(
            path_to_generated_data,
            gen_queries,
            corpus,
            gpl_steps,
            batch_size_gpl,
            cross_encoder,
            max_seq_length,
            qgen_prefix,
        )
        pseudo_labeler.run()
    # Do rescaling if needed:
    if rescale_range is not None and len(rescale_range) == 2:
        if gpl_score_function != "cos_sim":
            logger.warning(
                f"Doing rescaling while gpl_score_function = {gpl_score_function}"
            )

        new_min, new_max = rescale_range
        logger.info(f"Doing rescaling with new range [{new_min}, {new_max}]")
        gpl_training_data_fname = rescale_gpl_training_data(
            path_to_generated_data, new_min, new_max
        )  # This will rescale the margins and generate a new file
    else:
        # if len(rescale_range) != 2:
        #     logger.warning(f'len(rescale_range) should be 2')
        if gpl_score_function == "cos_sim":
            logger.warning(
                f"Not do rescaling while gpl_score_function = {gpl_score_function}"
            )

    ### Train the model with MarginMSE loss ###
    #### This will be skipped if the checkpoint at the indicated training steps can be found ####
    ckpt_dir = os.path.join(output_dir, str(gpl_steps))
    if not os.path.exists(ckpt_dir) or (
        os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)
    ):
        logger.info("Now doing training on the generated data with the MarginMSE loss")
        #### It can load checkpoints in both SBERT-format (recommended) and Huggingface-format
        model: SentenceTransformer = load_sbert(base_ckpt, pooling, max_seq_length)

        fpath_gpl_data = os.path.join(path_to_generated_data, gpl_training_data_fname)
        logger.info(f"Load GPL training data from {fpath_gpl_data}")
        train_dataset = GenerativePseudoLabelingDataset(
            fpath_gpl_data, gen_queries, corpus
        )
        train_dataloader = DataLoader(
            train_dataset, shuffle=False, batch_size=batch_size_gpl, drop_last=True
        )  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
        train_loss = MarginDistillationLoss(
            model=model, similarity_fct=gpl_score_function
        )

        # assert gpl_steps > 1000
        model.fit(
            [
                (train_dataloader, train_loss),
            ],
            epochs=1,
            steps_per_epoch=gpl_steps,
            warmup_steps=1000,
            checkpoint_save_steps=10000,
            checkpoint_save_total_limit=10000,
            output_path=output_dir,
            checkpoint_path=output_dir,
            use_amp=use_amp,
        )
    else:
        logger.info("Trained GPL model found. Now skip training")

    ### Evaluate the model if required ###
    if do_evaluation:
        logger.info("Doing evaluation for GPL")
        evaluate(
            evaluation_data,
            evaluation_output,
            ckpt_dir,
            max_seq_length,
            score_function=gpl_score_function,
            pooling=pooling,
            split=eval_split,
        )

    ### Train and evaluate QGen
    if mnrl_output_dir is not None:
        assert (
            mnrl_evaluation_output is not None
        ), "Evaluation path for MNRL should not be None, either"
        ckpt_dir = mnrl_output_dir
        if not os.path.exists(ckpt_dir) or (
            os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)
        ):
            logger.info("Now training MNRL on generated data")
            mnrl(
                path_to_generated_data,
                base_ckpt,
                mnrl_output_dir,
                max_seq_length,
                use_amp,
                qgen_prefix,
            )
        else:
            logger.info("Trained MNRL model found. Now skip training")

        logger.info("Doing evaluation for QGen (MNRL)")
        evaluate(
            evaluation_data,
            mnrl_evaluation_output,
            ckpt_dir,
            max_seq_length,
            score_function="cos_sim",
            pooling=pooling,
            split=eval_split,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_generated_data",
        required=True,
        help="Path for/to the generated data. GPL will first check this path for a `corpus.jsonl` file for the (sole) data input of the whole pipeline. If an empty folder is indicated, query generation and hard-negative mining will be run automatically; one can also use a BeIR-QGen format data folder to start and skip the query generation.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output path for the GPL model."
    )
    parser.add_argument(
        "--do_evaluation",
        action="store_true",
        default=False,
        help="Wether to do the evaluation (after training)",
    )
    parser.add_argument(
        "--evaluation_data",
        type=str,
        help="Path to the BeIR-format dataset. This is the next folder GPL goes to for the target corpus if there is no `corpus.jsonl` under `path_to_generated_data`",
    )
    parser.add_argument(
        "--evaluation_output", default="output", help="Path for the evaluation output."
    )
    parser.add_argument(
        "--qgen_prefix",
        default="qgen",
        help='This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.',
    )
    parser.add_argument(
        "--base_ckpt",
        default="distilbert-base-uncased",
        help="Initialization checkpoint in HF or SBERT format. Meaning-pooling will be used.",
    )
    parser.add_argument("--generator", default="BeIR/query-gen-msmarco-t5-base-v1")
    parser.add_argument(
        "--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--batch_size_gpl", type=int, default=32)
    parser.add_argument(
        "--batch_size_generation",
        type=int,
        default=10,
        help="Batch size in the query generation step.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=["cls", "mean", "max"],
        help="Specifying pooling method for dense retriever if in Huggingface-format. By default (None), it uses mean pooling. If in SBERT-format, there would be the indicated pooling method in its configure file and thus this argument will be ignored. ",
    )
    parser.add_argument("--max_seq_length", type=int, default=350)
    parser.add_argument(
        "--new_size",
        type=int,
        default=None,
        help="Resize the corpus to `new_size` (|corpus|) if needed. When set to None (by default), the |corpus| will be the full size. When set to -1, the |corpus| will be set automatically: If QPP * |corpus| <= 250K, |corpus| will be the full size; else QPP will be set 3 and |corpus| will be set to 250K / 3",
    )
    parser.add_argument(
        "--queries_per_passage",
        type=int,
        default=-1,
        help="Number of Queries Per Passage (QPP) in the query generation step. When set to -1 (by default), the QPP will be chosen automatically: If QPP * |corpus| <= 250K, then QPP will be set to 250K / |corpus|; else QPP will be set 3 and |corpus| will be set to 250K / 3",
    )
    parser.add_argument(
        "--gpl_steps", type=int, default=140000, help="Training steps for GPL."
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Whether to use half precision",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        help='Indicate retriever names for mining negatives. They could be one or many BM25 ("bm25") or dense retrievers (in SBERT format).',
    )
    parser.add_argument(
        "--retriever_score_functions",
        nargs="+",
        default=["cos_sim", "cos_sim"],
        choices=["dot", "cos_sim", "none"],
        help='Score functions of the corresponding retrievers for negative mining. Please set it to "none" for BM25.',
    )
    parser.add_argument(
        "--gpl_score_function", choices=["dot", "cos_sim"], default="dot"
    )
    parser.add_argument(
        "--rescale_range",
        nargs="+",
        type=float,
        default=None,
        help='Rescale the pseudo labels (i.e. score margins) to a certain range. For example, we can set this to "-2 2", which represents the margin range based on cosine-similarity. By default, it will not do rescaling.',
    )
    parser.add_argument(
        "--negatives_per_query",
        type=int,
        default=50,
        help="Mine how many negatives per query per retriever",
    )
    parser.add_argument("--mnrl_output_dir", default=None)
    parser.add_argument("--mnrl_evaluation_output", default=None)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="Which split to evaluate on",
    )
    parser.add_argument("--use_train_qrels", action="store_true", default=False)
    args = parser.parse_args()
    train(**vars(args))
