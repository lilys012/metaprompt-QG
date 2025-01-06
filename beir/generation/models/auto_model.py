from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm.autonotebook import trange
import torch, logging, math, queue
import torch.multiprocessing as mp
from typing import List, Dict

import os
import sys

import fire
import gradio as gr
import transformers
from peft import PeftModel
from transformers import GenerationConfig

from beir.generation.models.utils.callbacks import Iteratorize, Stream
from beir.generation.models.utils.prompter import Prompter

from tqdm import tqdm
import numpy as np
from simcse import SimCSE
import random
random.seed(123)

logger = logging.getLogger(__name__)


class FlanGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, dataset: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)
        self.dataset = dataset
    
    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None) -> List[str]:
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=350)
        
        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage  # 1
                    )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    num_return_sequences=ques_per_passage  # 1
                    )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)
 
    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=QGenModel._generate_multi_process_worker, args=(cuda_id, self.model, self.tokenizer, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}
    
    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()
    
    @staticmethod
    def _generate_multi_process_worker(target_device: str, model, tokenizer, input_queue, results_queue):
        """
        Internal working process to generate questions in multi-process setup
        """
        while True:
            try:
                id, batch_size, texts, ques_per_passage, top_p, top_k, max_length, dataset, gen_prefix = input_queue.get()
                model = model.to(target_device)
                generated_texts = []
                
                div = 1
                for start_idx in trange(0, len(texts), batch_size, desc='{}'.format(target_device)):
                    texts_batch = texts[start_idx:start_idx + batch_size]
                    div_texts_batch = []
                    encodings = tokenizer(texts_batch, padding=True, truncation=True, return_tensors="pt", max_length=350)
                    with torch.no_grad():
                        outs = model.generate(
                            input_ids=encodings['input_ids'].to(target_device), 
                            do_sample=True,
                            max_new_tokens=max_length,  # 64
                            top_k=top_k,  # 25,
                            top_p=top_p,
                            temperature=1.0,
                            repetition_penalty=1.0,
                            num_return_sequences=int(ques_per_passage/div)  # 1
                        )
                    outs_decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
                    generated_texts += outs_decoded
                
                results_queue.put([id, generated_texts])
            except queue.Empty:
                break
    
    def generate_multi_process(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_p: int, top_k: int, max_length: int, 
                               pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]

        if chunk_size is None:
            chunk_size = min(math.ceil(len(texts) / len(pool["processes"]) / 10), 5000)

        logger.info("Chunk data into packages of size {}, max {}".format(chunk_size, math.ceil(len(texts)/chunk_size)))

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for doc_text in texts:
            chunk.append(doc_text)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
            last_chunk_id += 1

        output_queue = pool['output']
        
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])        
        queries = [result[1] for result in results_list]
        
        return [item for sublist in queries for item in sublist]
        
class LlamaGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, dataset: str = None, **kwargs):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token='[PAD]'
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.model = self.model.bfloat16()

        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)
        self.dataset = dataset
    
    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None) -> List[str]:
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=350)
        
        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage  # 1
                    )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    num_return_sequences=ques_per_passage  # 1
                    )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)
  
    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=QGenModel._generate_multi_process_worker, args=(cuda_id, self.model, self.tokenizer, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}
    
    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()
    
    @staticmethod
    def _generate_multi_process_worker(target_device: str, model, tokenizer, input_queue, results_queue):
        """
        Internal working process to generate questions in multi-process setup
        """
        while True:
            try:
                id, batch_size, texts, ques_per_passage, top_p, top_k, max_length, dataset, gen_prefix = input_queue.get()
                model = model.to(target_device)
                generated_texts = []
                
                for start_idx in trange(0, len(texts), batch_size, desc='{}'.format(target_device)):
                    texts_batch = texts[start_idx:start_idx + batch_size]
                    div_texts_batch = []
                    encodings = tokenizer(texts_batch, padding=True, return_tensors="pt").to(target_device)
                    with torch.no_grad():
                        outs = model.generate(
                            # input_ids=torch.tensor(encodings).to(target_device), 
                            **encodings,
                            do_sample=True,
                            max_new_tokens=max_length,  # 64
                            top_k=top_k,  # 25,
                            top_p=top_p,
                            temperature=1.0,
                            repetition_penalty=1.0,
                            num_return_sequences=ques_per_passage 
                        )
                    outs_decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
                    outs_parsed = []
                    for i, od in enumerate(outs_decoded): 
                        query = od[len(texts_batch[i//ques_per_passage]):].strip()
                        try: query = query.split("\n")[0].strip()
                        except: pass
                        if query.find("Passage:") != -1: query = query[:query.find("Passage:")].strip()
                        outs_parsed.append(query)
                    generated_texts += outs_parsed
                
                results_queue.put([id, generated_texts])
            except queue.Empty:
                break
    
    def generate_multi_process(self, corpus: List[Dict[str, str]], corpus_ids: List[int], ques_per_passage: int, top_p: int, top_k: int, max_length: int, 
                               pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None, cid_to_query: Dict[str, str] = None):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        prompt = {"arguana":"Argument", "scifact":"Claim", "fever":"Claim", "scidocs":"Title", "dbpedia-entity":"Entity"}

        if cid_to_query != None:
            print("Encoding embeddings")
            embedding_model = SimCSE("models/sup-simcse-roberta-large")
            next_ice_num = 4
            truncated_docs = []
            embeddings = []
            for doc in tqdm(corpus):
                encodings = self.tokenizer(doc["title"] + " " + doc["text"], truncation=True, return_tensors="pt", max_length=350)
                decoded_text = self.tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
                truncated_docs.append(decoded_text)
                embeddings.append(embedding_model.encode(decoded_text).cpu().detach().numpy())
            embeddings = np.stack(embeddings)

            sim_scores_matrix = embeddings @ embeddings.T

        texts = []
        for i, doc in enumerate(tqdm(corpus)):
            if cid_to_query != None:

                sim_scores = sim_scores_matrix[i]
                sim_scores[i] = -12345
                sorted_indices = [i for i in range(len(sim_scores))]
                sorted_indices = sorted(sorted_indices, key = lambda x: -sim_scores[x])
                prompt_examples, query_examples = [], []
                for j in range(len(sorted_indices)):
                    try:
                        query_examples.append(cid_to_query[corpus_ids[sorted_indices[j]]])
                        prompt_examples.append(truncated_docs[sorted_indices[j]])
                    except: continue
                    if len(query_examples) >= next_ice_num: break
                prefix = ""
                for example, qex in zip(prompt_examples, query_examples):
                    prefix += "Passage: " + example + f"\n{prompt[self.dataset]}: " + qex + "\n\n"

                texts.append(prefix + "Passage: " + truncated_docs[i] + f"\n{prompt[self.dataset]}:")
            else:
                encodings = self.tokenizer(doc["title"] + " " + doc["text"], truncation=True, return_tensors="pt", max_length=350)
                decoded_text = self.tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
                if "Read the passage and generate a" in self.gen_prefix: #### baseline
                    texts.append("[INST] " + self.gen_prefix + f"[/INST] " + decoded_t ext + " Query:")
                else: #### few-shot
                    texts.append(self.gen_prefix + "Passage: " + decoded_text + f"\n{prompt[self.dataset]}:")

        if chunk_size is None:
            chunk_size = min(math.ceil(len(texts) / len(pool["processes"]) / 10), 5000)

        logger.info("Chunk data into packages of size {}, max {}".format(chunk_size, math.ceil(len(texts)/chunk_size)))

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for doc_text in texts:
            chunk.append(doc_text)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
            last_chunk_id += 1

        output_queue = pool['output']
        
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])        
        queries = [result[1] for result in results_list]
        
        return [item for sublist in queries for item in sublist]


class LlamaProtoGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, dataset: str = None, **kwargs):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token='[PAD]'
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.model = self.model.bfloat16()

        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)
        self.dataset = dataset
    
    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None) -> List[str]:
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=350)
        
        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage  # 1
                    )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_new_tokens=max_length,  # 64
                    top_k=top_k,  # 25,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    num_return_sequences=ques_per_passage  # 1
                    )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=QGenModel2._generate_multi_process_worker, args=(cuda_id, self.model, self.tokenizer, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}
    
    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()
    
    @staticmethod
    def _generate_multi_process_worker(target_device: str, model, tokenizer, input_queue, results_queue):
        """
        Internal working process to generate questions in multi-process setup
        """
        while True:
            try:
                id, batch_size, texts, ques_per_passage, top_p, top_k, max_length, dataset, gen_prefix = input_queue.get()
                model = model.to(target_device)
                generated_texts = []
                
                for start_idx in trange(0, len(texts), batch_size, desc='{}'.format(target_device)):
                    texts_batch = texts[start_idx:start_idx + batch_size]
                    div_texts_batch = []
                    encodings = tokenizer(texts_batch, padding=True, return_tensors="pt").to(target_device)
                    with torch.no_grad():
                        outs = model.generate(
                            **encodings,
                            do_sample=False,
                            max_new_tokens=max_length,  # 64
                            top_k=top_k,  # 25,
                            top_p=top_p,
                        )
                    outs_decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
                    outs_parsed = []
                    for i, od in enumerate(outs_decoded): 
                        query = od[len(texts_batch[i//ques_per_passage]):].strip()
                        try: query = query.split("\n")[0].strip()
                        except: pass
                        if query.find("Passage:") != -1: query = query[:query.find("Passage:")].strip()
                        outs_parsed.append(query)
                    generated_texts += outs_parsed
                
                results_queue.put([id, generated_texts])
            except queue.Empty:
                break
    
    def generate_multi_process(self, corpus: List[Dict[str, str]], corpus_ids: List[int], ques_per_passage: int, top_p: int, top_k: int, max_length: int, 
                               pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None, cid_to_query: Dict[str, str] = None):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        prompt = {"arguana":"Argument", "scifact":"Claim", "fever":"Claim", "scidocs":"Title", "dbpedia-entity":"Entity"}

        texts = []
        print("Transforming into prompts...")
        for i, doc in enumerate(tqdm(corpus)):
            encodings = self.tokenizer(doc["title"] + " " + doc["text"], truncation=True, return_tensors="pt", max_length=350)
            decoded_text = self.tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
            texts.append(f"[INST] Read the passage and generate a {prompt[self.dataset].lower()}. [/INST] " + decoded_text + f" {prompt[self.dataset]}:")

        if chunk_size is None:
            chunk_size = min(math.ceil(len(texts) / len(pool["processes"]) / 10), 5000)

        logger.info("Chunk data into packages of size {}, max {}".format(chunk_size, math.ceil(len(texts)/chunk_size)))

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for doc_text in texts:
            chunk.append(doc_text)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length, self.dataset, self.gen_prefix])
            last_chunk_id += 1

        output_queue = pool['output']
        
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])        
        queries = [result[1] for result in results_list]
        
        return [item for sublist in queries for item in sublist]