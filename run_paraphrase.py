# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for parapharase classification on Quora (DistilBERT, Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import sys
import argparse
import logging
import os
import re
import random
import glob
import timeit
import numpy as np
import json
import pickle
from functools import partial
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def load_data(path, mode):
    data_pair = []
    if mode == 'train' or mode == 'eval':
        assert type(path) == str
        with open(path) as f:
            data = json.load(f)
            for example in data:
                
                q1 = example['q1']
                q2 = example['q2']
                label = example['label']
                data_pair.append({'q1':q1, 'q2':q2, 'label':label})

    elif mode == 'score':
        assert type(path) == list and len(path) == 2
        with open(path[0],'r') as ref_file, open(path[1]) as hyp_file:
            for ref, hyp in zip(ref_file, hyp_file):
                data_pair.append({'q1':ref[:-1], 'q2':hyp[:-1]})
    return data_pair

class ParaphraseDataset(Dataset):
    def __init__(self, args, tokenizer, data, file_path, is_score=False, save_cache=True):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_%s'%filename.split('.')[0])

        if os.path.exists(cached_features_file) and not is_score:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from %s", file_path)

            self.examples = self._convert_to_features(data, tokenizer, is_score) 

            if save_cache:
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]['input']),\
               torch.tensor(self.examples[item]['token_type_id']), \
               torch.tensor(self.examples[item]['label']),\
               torch.tensor(self.examples[item]['example_id'])

    def _convert_to_features(self, data, tokenizer, is_score):
        examples = []
        for example_id, example in enumerate(tqdm(data, ncols=50)):
            try:
                q1 = tokenizer.encode(example['q1'])[1:-1]
                q2 = tokenizer.encode(example['q2'])[1:-1]
                label = example.get('label', 1)
            except KeyboardInterrupt:
                sys.exit()
            except: 
                print ('q1 : ', example['q1'])
                print ('q2 : ', example['q2'])
                continue

            # q1, q2
            inputs = [tokenizer.cls_token_id] + q1 +[tokenizer.sep_token_id] + q2
            token_type_id = [0]*(len(q1)+2) + [1]*(len(q2))
            assert len(inputs) == len(token_type_id)
            examples.append({'input':inputs, 'token_type_id':token_type_id, 'label':label,  'example_id':example_id})

            # q2, q1, change questions order to create more training data
            inputs = [tokenizer.cls_token_id] + q2 +[tokenizer.sep_token_id] + q1
            token_type_id = [0]*(len(q2)+2) + [1]*(len(q1))
            assert len(inputs) == len(token_type_id)
            examples.append({'input':inputs, 'token_type_id':token_type_id, 'label':label,  'example_id':example_id})
        return examples

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def my_collate(batch, pad_id):
    # padding sequence
    new_batch = []
    for i, elem in enumerate(zip(*batch)):
        if i == 0:
            new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=pad_id))
        elif i == 1:
            new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=0))
        else:
            new_batch.append(torch.stack(elem))

    return new_batch

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'summary'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    min_loss = np.inf
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], ncols=50)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids = batch[0]
            labels = batch[2]

            attention_mask = input_ids != tokenizer.pad_token_id

            inputs = {
                'input_ids':        input_ids,
                'attention_mask':   attention_mask,
                'labels':           labels,
            }

            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[1]  # XLM don't use segment_ids

            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            epoch_iterator.set_postfix(loss=loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_dataset)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info("eval_%s : %f"%(key, value))
                        min_loss = min(results['loss'], min_loss)

                    if not args.evaluate_during_training or results['loss'] == min_loss:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if (args.max_steps > 0 and global_step > args.max_steps):
                epoch_iterator.close()
                break
        if (args.max_steps > 0 and global_step > args.max_steps):
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, dataset, prefix=""):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                                 sampler=eval_sampler,
                                 collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id))

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()
    losses = []
    true_acc = []
    fake_acc = []
    
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids = batch[0]
        labels = batch[2]
        attention_mask = input_ids!=tokenizer.pad_token_id

        with torch.no_grad():
            inputs = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         labels,
            }
            
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[1]  # XLM don't use segment_ids

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})

            outputs = model(**inputs)

        logits = outputs[1]
        pred_class = torch.argmax(logits, dim=-1)
        acc = (pred_class == labels).cpu().numpy().astype(np.float)

        loss = outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training

        losses.append(loss.item())
        for label, a in zip(labels, acc):
            if label == 1:
                true_acc.append(a)
            else:
                fake_acc.append(a)
    losses = np.mean(losses)

    true_acc = np.mean(true_acc)
    fake_acc = np.mean(fake_acc)
    all_acc = (true_acc+fake_acc)/2
    results = {'loss':losses, 
               'acc':all_acc,
               'true_acc':true_acc, 'fake_acc':fake_acc}
    return results

def score(args, model, tokenizer, dataset, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                                 sampler=eval_sampler,
                                 collate_fn=partial(squad_collate, pad_id=tokenizer.pad_token_id))

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()
    all_acc = []
    results = defaultdict(list)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = [t.to(args.device) for t in batch]

        input_ids = batch[0]
        labels = batch[1]
        token_type_ids = batch[2]
        attention_mask = input_ids!=tokenizer.pad_token_id


        with torch.no_grad():
            inputs = {
                'input_ids':      input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels':         labels,
                'reduction':      'none'
            }
            
            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})

            outputs = model(**inputs)
        
        logits = outputs[1].cpu().numpy()
        pred_class = np.argmax(logits, -1)
        acc = (pred_class == labels.cpu().numpy()).astype(np.float)

        all_acc.extend(acc)

        probs = F.softmax(outputs[1], dim=-1)
        probs = probs[:,1].cpu().numpy().tolist()
        example_ids = batch[3].cpu().numpy().tolist()
        pred_ids = batch[4].cpu().numpy().tolist()

        for idx, p_idx, p in zip(example_ids, pred_ids, probs):
            if len(results[idx]) > p_idx:
                results[idx][p_idx] = max(results[idx][p_idx], p)
            else:
                results[idx].append(p)
    return results

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file_path", default=None, type=str,
                        help="")
    parser.add_argument("--eval_file_path", default=None, type=str,
                        help="")
    parser.add_argument("--score_file_path", default=None, type=str,
                        help="")
    parser.add_argument("--output_file_path", default=None, type=str,
                        help="")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_score", action='store_true',
                        help="Whether to run scoring.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--beam_size', type=float, default=1, help="")
    args = parser.parse_args()

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.num_labels=2
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_data = load_data(args.train_file_path, mode='train') 
        train_dataset = ParaphraseDataset(args, tokenizer, train_data, args.train_file_path)

        eval_data = load_data(args.eval_file_path, mode='eval') 
        eval_dataset = ParaphraseDataset(args, tokenizer, eval_data, args.eval_file_path)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = TextDataset(args, tokenizer, args.eval_file_path)
        checkpoints = [os.path.join(args.output_dir, 'checkpoint')]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, eval_dataset, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

        logger.info("Results: {}".format(results))
    
    if args.do_score:
        # read data
        score_dataset = SquadDataset(args, tokenizer, args.score_file_path,
                                     save_cache=False)
        # load checkpoint 
        checkpoint = os.path.join(args.output_dir, 'checkpoint')
        model = model_class.from_pretrained(checkpoint, force_download=True)
        model.to(args.device)

        # pred probs
        result = score(args, model, tokenizer, score_dataset)

        # write result
        with open(args.score_file_path, 'r') as f:
            data = json.load(f)
        for i, elem in enumerate(data):
            if result.get(i):
                elem['quora_prob'] = result[i]
        with open(args.output_file_path, 'w') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main()
