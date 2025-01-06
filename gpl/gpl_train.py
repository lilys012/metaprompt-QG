import gpl
import torch, gc
import argparse
gc.collect()
torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--qgen_prefix', required=True)
    parser.add_argument('--path_to_generated_data', required=True)
    parser.add_argument('--qrels_file', required=False, default="train")
    args = parser.parse_args()

    dataset = args.dataset
    qgen_prefix = args.qgen_prefix
    qrels_file = args.qrels_file

    ckpt = "msmarco-distilbert-base-tas-b"
    gpl.train(
        path_to_generated_data=args.path_to_generated_data,
        base_ckpt="sentence-transformers/msmarco-distilbert-base-tas-b",  
        gpl_score_function="dot",
        batch_size_gpl=32,
        gpl_steps=140000,
        new_size=-1,
        queries_per_passage=-1,
        output_dir=f"experiments/gpl/models/{dataset}/{ckpt}-{qgen_prefix}-{qrels_file}",
        evaluation_data=f"dataset/{dataset}",
        evaluation_output=f"experiments/gpl/eval/{dataset}/{ckpt}-{qgen_prefix}-{qrels_file}",
        retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["cos_sim", "cos_sim"],
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        qgen_prefix=qgen_prefix,
        do_evaluation=True,
        qrels_file=qrels_file,
    )